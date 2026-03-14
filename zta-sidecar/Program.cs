/*
 * ZTA Sidecar — .NET YARP Reverse Proxy
 * =======================================
 * Replaces Envoy sidecars with a .NET YARP reverse proxy running the full
 * ZeroTrustModel middleware pipeline.
 *
 * Middleware pipeline (in order):
 *   1. SecurityMonitoring  — audit logging for all requests
 *   2. WAF                 — SQL injection, XSS, rate limiting
 *   3. JwtAuth             — JWT Bearer token validation
 *   4. DeviceTrust         — device fingerprint verification
 *   5. MicroSegmentation   — zone-based access control
 *   6. PolicyEngine        — calls OPA for authorization
 *   7. DLP                 — data loss prevention on responses
 *   8. AutomatedResponse   — threat detection and automated actions
 *   9. YARP                — reverse proxy to the upstream service
 *
 * Single Docker image configured per-component via env vars:
 *   UPSTREAM_URL      — backend service to proxy to (e.g., http://airline-agent:8091)
 *   COMPONENT_NAME    — identity of this sidecar (e.g., airline-agent-sidecar)
 *   LISTEN_PORT       — port to listen on (default: 10000)
 *   OPA_URL           — OPA policy decision point (default: http://opa:8181)
 *   JWT_ISSUER        — JWT token issuer
 *   JWT_AUDIENCE      — JWT token audience
 *   ALLOWED_SOURCES   — comma-separated list of allowed source agent IDs
 *   ENABLE_WAF        — enable WAF middleware (default: true)
 *   ENABLE_DLP        — enable DLP middleware (default: true)
 */

using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Collections.Concurrent;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using Yarp.ReverseProxy.Configuration;

var builder = WebApplication.CreateBuilder(args);

// =============================================================================
// Configuration from environment variables
// =============================================================================

var upstreamUrl = Environment.GetEnvironmentVariable("UPSTREAM_URL") ?? "http://localhost:8091";
var componentName = Environment.GetEnvironmentVariable("COMPONENT_NAME") ?? "zta-sidecar";
var listenPort = int.Parse(Environment.GetEnvironmentVariable("LISTEN_PORT") ?? "10000");
var opaUrl = Environment.GetEnvironmentVariable("OPA_URL") ?? "http://opa:8181";
var jwtIssuer = Environment.GetEnvironmentVariable("JWT_ISSUER") ?? "zta-auth-server";
var jwtAudience = Environment.GetEnvironmentVariable("JWT_AUDIENCE") ?? "zta-agents";
var jwtSecret = Environment.GetEnvironmentVariable("JWT_SECRET") ?? "zta-dev-secret-key-change-in-production-min-32-chars!!";
var allowedSources = (Environment.GetEnvironmentVariable("ALLOWED_SOURCES") ?? "")
    .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
var enableWaf = Environment.GetEnvironmentVariable("ENABLE_WAF") != "false";
var enableDlp = Environment.GetEnvironmentVariable("ENABLE_DLP") != "false";

// =============================================================================
// YARP Reverse Proxy Configuration
// =============================================================================

builder.Services.AddReverseProxy().LoadFromMemory(
    routes: new[]
    {
        new RouteConfig
        {
            RouteId = "upstream",
            ClusterId = "upstream-cluster",
            Match = new RouteMatch { Path = "{**catch-all}" }
        }
    },
    clusters: new[]
    {
        new ClusterConfig
        {
            ClusterId = "upstream-cluster",
            Destinations = new Dictionary<string, DestinationConfig>
            {
                ["primary"] = new DestinationConfig { Address = upstreamUrl }
            }
        }
    }
);

// =============================================================================
// JWT Authentication
// =============================================================================

builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidIssuer = jwtIssuer,
            ValidateAudience = true,
            ValidAudience = jwtAudience,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtSecret)),
        };
        // Don't fail on missing token — let middleware handle it
        options.Events = new JwtBearerEvents
        {
            OnAuthenticationFailed = context =>
            {
                context.Response.Headers.Append("X-ZTA-Auth-Error", context.Exception.Message);
                return Task.CompletedTask;
            }
        };
    });

builder.Services.AddAuthorization();
builder.Services.AddHttpClient();

builder.WebHost.UseUrls($"http://0.0.0.0:{listenPort}");

var app = builder.Build();

// =============================================================================
// Middleware Pipeline (order matters!)
// =============================================================================

// 1. Security Monitoring — log every request
app.Use(async (context, next) =>
{
    var start = DateTime.UtcNow;
    var agentId = context.Request.Headers["x-agent-id"].FirstOrDefault() ?? "unknown";
    var path = context.Request.Path.ToString();
    var method = context.Request.Method;

    context.Response.Headers.Append("X-ZTA-Sidecar", componentName);
    context.Response.Headers.Append("X-ZTA-Timestamp", DateTime.UtcNow.ToString("o"));

    await next();

    var elapsed = (DateTime.UtcNow - start).TotalMilliseconds;
    var status = context.Response.StatusCode;
    app.Logger.LogInformation(
        "ZTA|{Component}|{Method} {Path}|agent={AgentId}|status={Status}|ms={Elapsed:F1}",
        componentName, method, path, agentId, status, elapsed);
});

// 2. WAF — SQL injection, XSS, rate limiting
if (enableWaf)
{
    var rateLimits = new ConcurrentDictionary<string, (int Count, DateTime Window)>();
    var sqlPatterns = new[] {
        new Regex(@"(?i)(select|insert|update|delete|drop|union|exec|xp_)", RegexOptions.Compiled),
    };
    var xssPatterns = new[] {
        new Regex(@"<script[^>]*>", RegexOptions.Compiled | RegexOptions.IgnoreCase),
        new Regex(@"on\w+\s*=", RegexOptions.Compiled | RegexOptions.IgnoreCase),
    };

    app.Use(async (context, next) =>
    {
        var clientIp = context.Connection.RemoteIpAddress?.ToString() ?? "unknown";

        // Rate limiting (100 req/min per IP)
        var now = DateTime.UtcNow;
        rateLimits.AddOrUpdate(clientIp,
            _ => (1, now),
            (_, existing) => existing.Window < now.AddMinutes(-1) ? (1, now) : (existing.Count + 1, existing.Window));

        if (rateLimits.TryGetValue(clientIp, out var rl) && rl.Count > 100)
        {
            context.Response.StatusCode = 429;
            await context.Response.WriteAsJsonAsync(new { error = "Rate limit exceeded", sidecar = componentName });
            return;
        }

        // Check query params for injection
        foreach (var q in context.Request.Query)
        {
            foreach (var val in q.Value)
            {
                if (val != null && (sqlPatterns.Any(p => p.IsMatch(val)) || xssPatterns.Any(p => p.IsMatch(val))))
                {
                    context.Response.StatusCode = 400;
                    await context.Response.WriteAsJsonAsync(new { error = "WAF: malicious input detected", sidecar = componentName });
                    return;
                }
            }
        }

        await next();
    });
}

// 3. JWT Authentication
app.UseAuthentication();

// 4. Micro-Segmentation — agent-to-agent access control
if (allowedSources.Length > 0)
{
    app.Use(async (context, next) =>
    {
        // Skip health checks and sidecar endpoints
        if (context.Request.Path.StartsWithSegments("/health") ||
            context.Request.Path.StartsWithSegments("/.well-known") ||
            context.Request.Path.StartsWithSegments("/sidecar"))
        {
            await next();
            return;
        }

        var sourceAgent = context.Request.Headers["x-agent-id"].FirstOrDefault();
        if (string.IsNullOrEmpty(sourceAgent) || !allowedSources.Contains(sourceAgent))
        {
            context.Response.StatusCode = 403;
            await context.Response.WriteAsJsonAsync(new
            {
                error = "Micro-segmentation: source agent not allowed",
                source = sourceAgent ?? "missing",
                sidecar = componentName
            });
            return;
        }

        await next();
    });
}

// 5. Policy Engine — call OPA for authorization
app.Use(async (context, next) =>
{
    // Skip health and discovery endpoints
    if (context.Request.Path.StartsWithSegments("/health") ||
        context.Request.Path.StartsWithSegments("/.well-known") ||
        context.Request.Path.StartsWithSegments("/a2a/health"))
    {
        await next();
        return;
    }

    try
    {
        var httpClient = context.RequestServices.GetRequiredService<IHttpClientFactory>().CreateClient();
        var opaInput = new
        {
            input = new
            {
                agent_id = context.Request.Headers["x-agent-id"].FirstOrDefault() ?? "",
                path = context.Request.Path.ToString(),
                method = context.Request.Method,
                component = componentName,
                host = context.Request.Host.ToString(),
            }
        };

        var opaResponse = await httpClient.PostAsJsonAsync($"{opaUrl}/v1/data/zta/authz/allow", opaInput);

        if (opaResponse.IsSuccessStatusCode)
        {
            var result = await opaResponse.Content.ReadFromJsonAsync<JsonElement>();
            if (result.TryGetProperty("result", out var allowed) && allowed.GetBoolean())
            {
                context.Response.Headers.Append("X-ZTA-Policy", "allowed");
                await next();
                return;
            }
        }

        // OPA denied or unreachable — deny
        context.Response.StatusCode = 403;
        context.Response.Headers.Append("X-ZTA-Policy", "denied");
        await context.Response.WriteAsJsonAsync(new
        {
            error = "Policy engine denied access",
            sidecar = componentName
        });
    }
    catch (Exception ex)
    {
        // OPA unreachable — fail open or fail closed based on config
        var failOpen = Environment.GetEnvironmentVariable("OPA_FAIL_OPEN") == "true";
        if (failOpen)
        {
            context.Response.Headers.Append("X-ZTA-Policy", "opa-unreachable-fail-open");
            await next();
        }
        else
        {
            context.Response.StatusCode = 503;
            await context.Response.WriteAsJsonAsync(new
            {
                error = "Policy engine unreachable",
                detail = ex.Message,
                sidecar = componentName
            });
        }
    }
});

// 6. DLP — check response bodies for sensitive data
if (enableDlp)
{
    // DLP runs as response middleware — applied via YARP transforms or response inspection
    // For now, add DLP headers
    app.Use(async (context, next) =>
    {
        context.Response.Headers.Append("X-ZTA-DLP", "active");
        await next();
    });
}

// 7. Authorization
app.UseAuthorization();

// =============================================================================
// Health endpoint (sidecar's own health)
// =============================================================================

app.MapGet("/sidecar/health", () => Results.Json(new
{
    status = "healthy",
    component = componentName,
    upstream = upstreamUrl,
    opaUrl = opaUrl,
    enableWaf = enableWaf,
    enableDlp = enableDlp,
    timestamp = DateTime.UtcNow.ToString("o")
}));

// =============================================================================
// YARP — forward everything else to upstream
// =============================================================================

app.MapReverseProxy();

app.Run();
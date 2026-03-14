# =============================================================================
# ZTA Policy for .NET YARP Sidecar Integration
# =============================================================================
# OPA Policy Decision Point — called by sidecar PolicyEngine middleware
# POST /v1/data/zta/authz/allow  { "input": { agent_id, path, method, component, host } }
# =============================================================================

package zta.authz

import future.keywords.if
import future.keywords.in

# Default deny
default allow := false

# =============================================================================
# Agent Registry — defines allowed communication paths
# =============================================================================

agent_registry := {
    "supervisor-agent": {
        "type": "supervisor",
        "allowed_targets": [
            "airline-agent", "hotel-agent", "car-rental-agent"
        ]
    },
    "airline-agent": {
        "type": "worker",
        "domain": "airline",
        "allowed_targets": ["airline-mcp"]
    },
    "hotel-agent": {
        "type": "worker",
        "domain": "hotel",
        "allowed_targets": ["hotel-mcp"]
    },
    "car-rental-agent": {
        "type": "worker",
        "domain": "car-rental",
        "allowed_targets": ["car-rental-mcp"]
    }
}

# Component name to service mapping
component_to_service := {
    "airline-agent-sidecar": "airline-agent",
    "hotel-agent-sidecar": "hotel-agent",
    "car-rental-agent-sidecar": "car-rental-agent",
    "airline-mcp-sidecar": "airline-mcp",
    "hotel-mcp-sidecar": "hotel-mcp",
    "car-rental-mcp-sidecar": "car-rental-mcp",
    "supervisor-sidecar": "supervisor-agent"
}

# =============================================================================
# Input extraction
# =============================================================================

agent_id := input.agent_id
request_path := input.path
request_method := input.method
component := input.component

# Resolve the target service from the sidecar component name
target_service := service if {
    service := component_to_service[component]
} else := component

# =============================================================================
# Allow Rules
# =============================================================================

# Always allow health checks
allow if {
    request_path == "/health"
}

# Always allow A2A health
allow if {
    request_path == "/a2a/health"
}

# Always allow Agent Card discovery
allow if {
    request_path == "/.well-known/agent.json"
    request_method == "GET"
}

# Always allow sidecar health
allow if {
    startswith(request_path, "/sidecar/")
}

# Allow registered agents calling allowed targets
allow if {
    agent_registry[agent_id]
    agent := agent_registry[agent_id]
    target_service in agent.allowed_targets
}

# Allow supervisor to access A2A endpoints on worker agents
allow if {
    agent_id == "supervisor-agent"
    request_path == "/a2a"
}

# Allow supervisor to access agent card endpoints
allow if {
    agent_id == "supervisor-agent"
    request_path == "/.well-known/agent.json"
}

# Allow workers to access MCP endpoints
allow if {
    agent_registry[agent_id]
    agent := agent_registry[agent_id]
    agent.type == "worker"
    startswith(request_path, "/sse")
}

allow if {
    agent_registry[agent_id]
    agent := agent_registry[agent_id]
    agent.type == "worker"
    startswith(request_path, "/mcp")
}

# =============================================================================
# Deny reasons (for debugging)
# =============================================================================

deny_reason := "agent not registered" if {
    not agent_registry[agent_id]
}

deny_reason := "target not allowed for agent" if {
    agent_registry[agent_id]
    agent := agent_registry[agent_id]
    not target_service in agent.allowed_targets
}

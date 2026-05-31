# Prompt fragments

Agent methods accept `prompt_fragment` to add extra instruction text during loop construction.

```python
response = client.run_agent(
    user_input="Analyze this incident.",
    tools=[lookup_runbook],
    prompt_fragment="Always cite the runbook section used.",
)
```

`prompt_fragment` may be a static string or a callable returning a string.

```python
def fragment():
    return "Prefer safe, reversible actions before risky ones."

response = client.run_agent(
    user_input="Help with deployment triage.",
    tools=[get_deploy_status],
    prompt_fragment=fragment,
)
```

Use prompt fragments for cross-cutting instructions that should not be baked into every user message, such as safety rules, citation style, or per-iteration operating constraints.

Keep fragments short. Long fragments consume context repeatedly and can dilute the user's task.

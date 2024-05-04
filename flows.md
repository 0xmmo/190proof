# flows

## multi-step

- graph of tools available
- if x is called, then sub tools a,b,c are next

```py
def decide(context, tools):
    tool, args = llm.choose_tool(context, tools)
    if tool.is_final():
        return args.response
    else:
        result = tool.execute(context, args)
        context.add(tool_call, result)
        decide(context, tool.sub_tools)

```

## recursive

- use same set of tools until done

```py
def decide(context, tools):
    tool, args = llm.choose_tool(context, tools)
    if tool.is_final():
        return args.response
    else:
        result = tool.execute(context, args)
        context.add(tool_call, result)
        decide(context, tools)
```

- recursion limit
- at least one tool must be final
- configurability, e.g. change context or limit tools after specific tool is called
- tool result can be a simple { success: true }

## combined

- simple

```py
def decide(context, tools, max_depth):
    tool, args = llm.choose_tool(context, tools)
    if tool.is_final:
        return args.response
    else:
        result = tool.execute(context, args)
        context.add(tool_call, result)
        if tool.sub_tools:
            decide(context, sub_tools)
        else:
            decide(context, tools)
```

- error handling

```py
def decide(context, tools, max_depth):
    if every tools is not is_final:
        throw('No terminating tool provided')
    if len(context) > max_depth:
        throw('Max depth exceeded')

    tool, args = llm.choose_tool(context, tools)
    if tool.is_final:
        return args.response
    else:
        result = tool.execute(context, args)
        context.add(tool_call, result)
        if tool.sub_tools:
            decide(context, sub_tools)
        else:
            decide(context, tools)
```

## tools:

- callResearchExpert
- callRemindersExpert
  - setReminder {is_final:true}
  - deleteReminder {is_final:true}

import {
  ClaudeModel,
  GPTModel,
  OpenAIPayload,
  OpenAIMessage,
  OpenAIConfig,
  AnthropicAIPayload,
  AnthropicAIMessage,
  GenericMessage,
  AnthropicAIConfig,
  GenericPayload,
  GroqPayload,
  GroqModel,
  ParsedResponseMessage,
  FunctionCall,
} from "./interfaces";
import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import axios from "axios";
import { timeout } from "./utils";

export { ClaudeModel, GPTModel, GroqModel, OpenAIConfig } from "./interfaces";

function parseStreamedResponse(
  identifier: string,
  paragraph: string,
  functionCallName: string,
  functionCallArgs: string,
  allowedFunctionNames: Set<string> | null
): ParsedResponseMessage {
  let functionCall: ParsedResponseMessage["function_call"] = null;
  if (functionCallName && functionCallArgs) {
    if (allowedFunctionNames && !allowedFunctionNames.has(functionCallName)) {
      throw new Error(
        "Stream error: received function call with unknown name: " +
          functionCallName
      );
    }

    functionCall = {
      name: functionCallName,
      arguments: JSON.parse(functionCallArgs), // allow JSON.parse to throw
    };
  }

  if (!paragraph && !functionCall) {
    console.error(
      identifier,
      "Stream error: received message without content or function_call, raw:",
      JSON.stringify({ paragraph, functionCallName, functionCallArgs })
    );
    throw new Error(
      "Stream error: received message without content or function_call"
    );
  }

  return {
    role: "assistant",
    content: paragraph || null,
    function_call: functionCall,
  };
}

async function callOpenAiWithRetries(
  identifier: string,
  openAiPayload: OpenAIPayload,
  openAiConfig?: OpenAIConfig,
  retries: number = 5,
  chunkTimeoutMs: number = 15_000
): Promise<ParsedResponseMessage> {
  console.log(
    identifier,
    "Calling OpenAI API with retries:",
    openAiConfig?.service,
    openAiPayload.model,
    openAiConfig?.baseUrl
  );

  let errorObj: any;
  for (let i = 0; i <= retries; i++) {
    try {
      const timerId = `timer:${identifier}:${Date.now()}:callOpenAi:${
        openAiConfig?.service
      }-${openAiPayload.model}-${openAiConfig?.orgId}`;
      const res = await callOpenAIStream(
        identifier,
        openAiPayload,
        openAiConfig,
        chunkTimeoutMs
      );
      return res;
    } catch (error: any) {
      const errorCode = error.data?.code;

      if (errorCode) {
        console.error(
          identifier,
          `Retry #${i} failed with API error: ${errorCode}`,
          JSON.stringify({
            data: error.data,
          })
        );
      }

      // to solve context length issue or JSON parsing error due to truncated response
      openAiPayload.model = GPTModel.GPT4_1106_PREVIEW;
      openAiPayload.temperature = 0.8; // Higher temperature

      // on 2nd or more retries
      // if Azure content policy error is persistent
      if (
        i >= 2 &&
        openAiConfig?.service === "azure" &&
        errorCode === "content_filter"
      ) {
        console.log(
          identifier,
          `Switching to OpenAI service due to content filter error`
        );
        openAiConfig.service = "openai"; // Move to OpenAI, failed due to Azure content policy
      }

      // on 3rd retry
      if (i === 3) {
        if (openAiConfig?.service === "azure") {
          openAiConfig.service = "openai";
        }
      }

      // on 4th retry
      if (i === 4) {
        // abort function calling, e.g. stubborn `python` function call case
        if (openAiPayload.function_call) {
          openAiPayload.function_call = "none";
        }
      }

      console.error(
        identifier,
        `Retrying due to error: received bad response from OpenAI API [${
          openAiConfig?.service
        }-${openAiPayload.model}-${openAiConfig?.orgId}]: ${
          error.message
        } - ${JSON.stringify(error.response?.data)}`
      );

      await timeout(250);
    }
  }

  console.error(
    identifier,
    `Failed to call OpenAI API after ${retries} attempts. Please lookup OpenAI status for active issues.`,
    errorObj
  );
  throw new Error(
    `${identifier}: Failed to call OpenAI API after ${retries} attempts. Please lookup OpenAI status for active issues.`
  );
}

async function callOpenAIStream(
  identifier: string,
  openAiPayload: OpenAIPayload,
  openAiConfig: OpenAIConfig | undefined,
  chunkTimeoutMs: number
): Promise<ParsedResponseMessage> {
  const functionNames: Set<string> | null = openAiPayload.functions
    ? new Set(openAiPayload.functions.map((fn) => fn.name as string))
    : null;

  if (!openAiConfig) {
    const defaultOpenAIBaseUrl = // TODO: Remove this one we have per-provider configs
      "https://gateway.ai.cloudflare.com/v1/932636fc124abb5171fd630afe668905/igpt";
    openAiConfig = {
      service: "openai",
      apiKey: process.env.OPENAI_API_KEY as string,
      baseUrl: defaultOpenAIBaseUrl,
    };
  }

  console.log("payload", openAiPayload);

  let response;
  const controller = new AbortController();
  if (openAiConfig.service === "azure") {
    console.log(identifier, "Using Azure OpenAI service", openAiPayload.model);
    const model = openAiPayload.model;

    if (!openAiConfig.modelConfigMap) {
      throw new Error(
        "OpenAI config modelConfigMap is required when using Azure OpenAI service."
      );
    }

    const azureConfig = openAiConfig.modelConfigMap[model];
    const endpoint = `${openAiConfig.baseUrl}/azure-openai/${azureConfig.resource}/${azureConfig.deployment}/chat/completions?api-version=${azureConfig.apiVersion}`;

    response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "api-key": azureConfig.apiKey,
      },
      body: JSON.stringify({
        ...openAiPayload,
        stream: true,
      }),
      signal: controller.signal,
    });
  } else {
    // openai by default
    console.log(identifier, "Using OpenAI service", openAiPayload.model);
    const endpoint = `${openAiConfig.baseUrl}/openai/chat/completions`;
    if (openAiConfig.orgId) {
      console.log(identifier, "Using orgId", openAiConfig.orgId);
    }

    response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${openAiConfig.apiKey}`,
        ...(openAiConfig.orgId
          ? { "OpenAI-Organization": openAiConfig.orgId }
          : {}),
      },
      body: JSON.stringify({
        ...openAiPayload,
        stream: true,
      }),
      signal: controller.signal,
    });
  }

  if (response.body) {
    let paragraph = "";
    let functionCallName = "";
    let functionCallArgs = "";

    const reader = response.body.getReader();

    let partialChunk = "";
    let abortTimeout: NodeJS.Timeout;
    const startAbortTimeout = () => {
      clearTimeout(abortTimeout);
      return setTimeout(() => {
        console.log(
          identifier,
          `Stream error: aborted due to timeout after ${chunkTimeoutMs} ms.`,
          JSON.stringify({ paragraph })
        );
        controller.abort();
      }, chunkTimeoutMs);
    };

    let chunkIndex = -1;
    while (true) {
      chunkIndex++;
      const abortTimeout = startAbortTimeout();
      const { done, value } = await reader.read();
      clearTimeout(abortTimeout);

      if (done) {
        console.log(
          identifier,
          `Stream error: ended after ${
            chunkIndex + 1
          } chunks via reader done flag.`,
          paragraph,
          functionCallName,
          functionCallArgs
        );
        throw new Error("Stream error: ended prematurely");
      }

      let chunk = new TextDecoder().decode(value);
      if (partialChunk) {
        chunk = partialChunk + chunk;
        partialChunk = "";
      }
      let jsonStrings = chunk.split(/^data: /gm);

      for (let jsonString of jsonStrings) {
        if (!jsonString) {
          continue;
        }

        if (jsonString.includes("[DONE]")) {
          console.log(
            identifier,
            `Stream explicitly marked as done after ${chunkIndex + 1} chunks.`
          );
          return parseStreamedResponse(
            identifier,
            paragraph,
            functionCallName,
            functionCallArgs,
            functionNames
          );
        }

        let json;
        try {
          json = JSON.parse(jsonString.trim());
        } catch (error: any) {
          partialChunk = jsonString; // We're assuming any JSON parsing error means we got a non-terminated JSON for a chunk
          continue;
        }

        if (!json.choices || !json.choices.length) {
          if (json.error) {
            console.error(
              identifier,
              "Stream error: OpenAI error:",
              json.error
            );
            const error = new Error("Stream error: OpenAI error") as any;
            error.data = json.error;
            throw error;
          }
          if (chunkIndex !== 0)
            console.error(
              identifier,
              "Stream error: no choices in JSON:",
              json
            ); // bad if it's not the first chunk
          continue;
        }

        const dFn = json.choices[0].delta.function_call;
        if (dFn) {
          if (dFn.name) functionCallName += dFn.name;
          if (dFn.arguments) functionCallArgs += dFn.arguments;
        }

        const text = json.choices[0].delta.content;
        if (text) {
          paragraph += text;
        }
      }
    }
  } else {
    throw new Error("Stream error: no response body");
  }
}

async function callAnthropicWithRetries(
  identifier: string,
  AiPayload: AnthropicAIPayload,
  AiConfig?: AnthropicAIConfig,
  attempts = 5
): Promise<ParsedResponseMessage> {
  console.log(identifier, "Calling Anthropic API with retries");
  let lastResponse;
  for (let i = 0; i < attempts; i++) {
    // if last attempt
    if (i === attempts - 1) {
      AiPayload.model = ClaudeModel.SONNET; // fallback to sonnet model
    }

    try {
      // return await callAnthropic(identifier, AiPayload);
      lastResponse = await callAnthropic(identifier, AiPayload, AiConfig);
      return lastResponse;
    } catch (e: any) {
      console.error(
        identifier,
        `Retrying due to error: received bad response from Anthropic API: ${e.message}`,
        JSON.stringify(e.response?.data)
      );

      if (e.response?.data?.error?.type === "rate_limit_error") {
        // upgrade to Sonnet
        AiPayload.model = ClaudeModel.SONNET;
      }

      await timeout(125 * i);
    }
  }
  const error = new Error(
    `Failed to call Anthropic API after ${attempts} attempts`
  ) as any;
  error.response = lastResponse;
  throw error;
}

async function callAnthropic(
  identifier: string,
  AiPayload: AnthropicAIPayload,
  AiConfig?: AnthropicAIConfig
): Promise<ParsedResponseMessage> {
  const anthropicMessages = jigAnthropicMessages(AiPayload.messages);

  let data;
  let response;
  if (AiConfig?.service === "bedrock") {
    // DOES NOT SUPPORT TOOLS YET
    const AWS_REGION = "us-east-1";
    const MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0";

    // set in environment
    // process.env.AWS_ACCESS_KEY_ID = AWS_ACCESS_KEY_ID;
    // process.env.AWS_SECRET_ACCESS_KEY = AWS_SECRET_ACCESS_KEY;

    const client = new BedrockRuntimeClient({ region: AWS_REGION });
    const payload = {
      anthropic_version: "bedrock-2023-05-31",
      max_tokens: 4096,
      messages: anthropicMessages,
      tools: AiPayload.functions?.map((f) => ({
        ...f,
        input_schema: f.parameters,
        parameters: undefined,
      })),
    };

    const response = await client.send(
      new InvokeModelCommand({
        contentType: "application/json",
        body: JSON.stringify(payload),
        modelId: MODEL_ID,
      })
    );

    const decodedResponseBody = new TextDecoder().decode(response.body);
    data = JSON.parse(decodedResponseBody);
  } else {
    // default to anthropic
    const response = await axios.post(
      "https://api.anthropic.com/v1/messages",
      {
        model: AiPayload.model,
        messages: anthropicMessages,
        tools: AiPayload.functions?.map((f) => ({
          ...f,
          input_schema: f.parameters,
          parameters: undefined,
        })),
        max_tokens: 4096,
      },
      {
        headers: {
          "content-type": "application/json",
          "x-api-key": process.env.ANTHROPIC_API_KEY as string,
          "anthropic-version": "2023-06-01",
          "anthropic-beta": "tools-2024-04-04",
        },
        timeout: 60000,
      }
    );

    data = response.data;
  }

  const answers = data.content;

  if (!answers[0]) {
    console.error(identifier, "Missing answer in Anthropic API:", data);
    throw new Error("Missing answer in Anthropic API");
  }

  let textResponse = "";
  let functionCalls: any[] = [];
  for (const answer of answers) {
    if (!answer.type) {
      console.error(identifier, "Missing answer type in Anthropic API:", data);
      throw new Error("Missing answer type in Anthropic API");
    }

    let text = "";
    if (answer.type === "text") {
      text = answer.text
        .replace(/<thinking>.*?<\/thinking>/gs, "")
        .replace(/<answer>|<\/answer>/gs, "")
        .trim();

      if (!text) {
        // remove the tags and return the text within
        text = answer.text.replace(
          /<thinking>|<\/thinking>|<answer>|<\/answer>/gs,
          ""
        );
        console.log("No text in answer, returning text within tags:", text);
      }

      if (textResponse) {
        textResponse += `\n\n${text}`;
      } else {
        textResponse = text;
      }
    } else if (answer.type === "tool_use") {
      const call = {
        name: answer.name,
        arguments: answer.input,
      };
      functionCalls.push(call);
    }
  }

  if (!textResponse && !functionCalls.length) {
    console.error(
      identifier,
      "Missing text & fns in Anthropic API response:",
      JSON.stringify(data)
    );
    throw new Error("Missing text & fns in Anthropic API response");
  }

  return {
    role: "assistant",
    content: textResponse,
    function_call: functionCalls[0],
  };
}

function jigAnthropicMessages(
  messages: AnthropicAIMessage[]
): AnthropicAIMessage[] {
  // Takes a list if messages each with a role and content

  // First, changes the role of every system message to "user"
  let jiggedMessages = messages.map((message) => {
    if (message.role === "system") {
      return {
        role: "user" as const,
        content: `# CONTEXT ---\n${message.content}\nBefore answering you can reason about the instructions and answer using <thinking></thinking> tags\n---`,
      };
    }
    return message;
  });

  // If the first message is not user, add an empty user message at the start
  if (jiggedMessages[0]?.role !== "user") {
    jiggedMessages = [
      {
        role: "user" as const,
        content: "...",
      },
      ...jiggedMessages,
    ];
  }

  // Then, makes sure that user and assistant messages alternate, where they do not, insert a message with empty content to make them alternate
  jiggedMessages = jiggedMessages.reduce((acc, message) => {
    if (acc.length === 0) {
      return [message];
    }

    const lastMessage = acc[acc.length - 1];
    if (lastMessage.role === message.role) {
      return [
        ...acc,
        {
          role:
            message.role === "user"
              ? ("assistant" as const)
              : ("user" as const),
          content: "...",
        },
        message,
      ];
    }

    return [...acc, message];
  }, [] as AnthropicAIMessage[]);

  // If last message in array is assistant, then add an empty user message
  if (jiggedMessages[jiggedMessages.length - 1]?.role === "assistant") {
    jiggedMessages.push({
      role: "user",
      content: "...",
    });
  }

  return jiggedMessages;
}

export async function callWithRetries(
  identifier: string,
  aiPayload: GenericPayload,
  aiConfig?: OpenAIConfig | AnthropicAIConfig,
  retries: number = 5,
  chunkTimeoutMs: number = 15_000
): Promise<ParsedResponseMessage> {
  // Determine which service to use based on the model type
  if (isAnthropicPayload(aiPayload)) {
    console.log(identifier, "Delegating call to Anthropic API");

    return await callAnthropicWithRetries(
      identifier,
      prepareAnthropicPayload(aiPayload),
      aiConfig as AnthropicAIConfig,
      retries
    );
  } else if (isOpenAiPayload(aiPayload)) {
    console.log(identifier, "Delegating call to OpenAI API");
    return await callOpenAiWithRetries(
      identifier,
      prepareOpenAIPayload(aiPayload),
      aiConfig as OpenAIConfig,
      retries,
      chunkTimeoutMs
    );
  } else if (isGroqPayload(aiPayload)) {
    console.log(identifier, "Delegating call to Groq API");
    return await callGroqWithRetries(identifier, prepareGroqPayload(aiPayload));
  } else {
    throw new Error("Invalid AI payload: Unknown model type.");
  }
}

function isAnthropicPayload(payload: any): Boolean {
  return Object.values(ClaudeModel).includes(payload.model);
}

function prepareAnthropicPayload(payload: GenericPayload): AnthropicAIPayload {
  return {
    model: payload.model as ClaudeModel,
    messages: payload.messages.map((message) => ({
      role: message.role,
      content: message.content,
      // TODO: Handle files
    })),
    functions: payload.functions,
  };
}

function isOpenAiPayload(payload: any): Boolean {
  return Object.values(GPTModel).includes(payload.model);
}

function prepareOpenAIPayload(payload: GenericPayload): OpenAIPayload {
  return {
    model: payload.model as GPTModel,
    messages: payload.messages.map((message) => ({
      role: message.role,
      content: normalizeMessageContent(message.content),
      // TODO: Handle files
    })),
    functions: payload.functions,
    function_call: payload.function_call,
  };
}

function isGroqPayload(payload: any): Boolean {
  return Object.values(GroqModel).includes(payload.model);
}

function prepareGroqPayload(payload: GenericPayload): GroqPayload {
  return {
    model: payload.model as GroqModel,
    messages: payload.messages.map((message) => ({
      role: message.role,
      content: normalizeMessageContent(message.content),
    })),
    tools: payload.functions?.map((fn) => ({
      type: "function",
      function: fn,
    })),
    tool_choice: payload.function_call
      ? typeof payload.function_call === "string"
        ? payload.function_call // "none" | "auto"
        : {
            type: "function",
            function: payload.function_call,
          }
      : undefined,
    temperature: payload.temperature,
  };
}

function normalizeMessageContent(
  content: AnthropicAIMessage["content"]
): string {
  return Array.isArray(content)
    ? content
        .map((c) => (c.type === "text" ? c.text : `[${c.type}]`))
        .join("\n")
    : content;
}

async function callGroq(
  identifier: string,
  payload: GroqPayload
): Promise<ParsedResponseMessage> {
  const response = await axios.post(
    "https://api.groq.com/openai/v1/chat/completions",
    payload,
    {
      headers: {
        "content-type": "application/json",
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      },
    }
  );

  const data = response.data;

  const answer = data.choices[0].message;
  if (!answer) {
    console.error(identifier, "Missing answer in Groq API:", data);
    throw new Error("Missing answer in Groq API");
  }

  const textResponse = answer.content || null;
  let functionCall: FunctionCall | null = null;
  if (answer.tool_calls && answer.tool_calls.length) {
    const toolCall = answer.tool_calls[0];
    functionCall = {
      name: toolCall.function.name,
      arguments: JSON.parse(toolCall.function.arguments),
    };
  }

  return {
    role: "assistant",
    content: textResponse,
    function_call: functionCall,
  };
}

async function callGroqWithRetries(
  identifier: string,
  payload: GroqPayload,
  retries: number = 5
): Promise<ParsedResponseMessage> {
  console.log(identifier, "Calling Groq API with retries");

  let lastResponse;
  for (let i = 0; i < retries; i++) {
    try {
      lastResponse = await callGroq(identifier, payload);
      return lastResponse;
    } catch (e: any) {
      console.error(
        identifier,
        `Retrying due to error: received bad response from Groq API: ${e.message}`,
        JSON.stringify(e.response?.data)
      );

      await timeout(125 * i);
    }
  }
  const error = new Error(
    `Failed to call Groq API after ${retries} attempts`
  ) as any;
  error.response = lastResponse;
  throw error;
}

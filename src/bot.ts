import './fetch-polyfill'

import {info, setFailed, warning} from '@actions/core'
import {OpenAI} from 'openai' // Fixed import
import pRetry from 'p-retry'
import {OpenAIOptions, Options} from './options'

// define type to save messageId and threadId
export interface Ids {
  messageId?: string
  threadId?: string
}

export interface ChatMessage {
  id: string
  content: string
  role: 'assistant' | 'user' | 'system'
}

export class Bot {
  private readonly client: OpenAI | null = null
  private readonly options: Options
  private readonly systemMessage: string

  constructor(options: Options, openaiOptions: OpenAIOptions) {
    this.options = options
    if (process.env.OPENAI_API_KEY) {
      const currentDate = new Date().toISOString().split('T')[0]
      this.systemMessage = `${options.systemMessage} 
Knowledge cutoff: ${openaiOptions.tokenLimits.knowledgeCutOff}
Current date: ${currentDate}

IMPORTANT: Entire response must be in the language with ISO code: ${options.language}
`

      this.client = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY,
        organization: process.env.OPENAI_API_ORG,
        baseURL: options.apiBaseUrl || undefined,
        timeout: options.openaiTimeoutMS,
        maxRetries: options.openaiRetries
      })
    } else {
      const err =
        "Unable to initialize the OpenAI API, 'OPENAI_API_KEY' environment variable is not available"
      throw new Error(err)
    }
  }

  chat = async (message: string, ids: Ids): Promise<[string, Ids]> => {
    let res: [string, Ids] = ['', {}]
    try {
      res = await this.chat_(message, ids)
      return res
    } catch (e: unknown) {
      if (e instanceof Error) {
        warning(`Failed to chat: ${e.message}, backtrace: ${e.stack}`)
      }
      return res
    }
  }

  private readonly chat_ = async (
    message: string,
    ids: Ids
  ): Promise<[string, Ids]> => {
    const start = Date.now()
    if (!message || !this.client) {
      return ['', {}]
    }

    try {
      const messages = [
        {
          role: 'system' as const,
          content: this.systemMessage
        },
        {
          role: 'user' as const,
          content: message
        }
      ]

      // Add message history if available
      if (ids.messageId && ids.threadId) {
        try {
          const previousMessages = await this.client.beta.threads.messages.list(
            ids.threadId
          )
          for (const msg of previousMessages.data) {
            messages.push({
              role: msg.role as 'system' | 'user',
              content: msg.content[0].type === 'text' 
                ? msg.content[0].text.value 
                : ''
            })
          }
        } catch (error) {
          warning('Failed to retrieve message history, continuing with new message')
        }
      }

      const response = await pRetry(
        async () => {
          const completion = await this.client!.chat.completions.create({
            messages,
            model: this.options.openaiHeavyModel || 'o3-mini', // Type assertion
            temperature: this.options.openaiModelTemperature,
            max_tokens: this.options.heavyTokenLimits.maxTokens,
            stream: false
          })

          return completion.choices[0].message
        },
        {
          retries: this.options.openaiRetries,
          onFailedAttempt: error => {
            info(
              `Attempt failed: ${error.message}. ${error.attemptNumber} of ${
                this.options.openaiRetries + 1
              } attempts made.`
            )
          }
        }
      )

      const end = Date.now()
      info(`openai response time: ${end - start} ms`)

      if (this.options.debug) {
        info(`openai response: ${JSON.stringify(response)}`)
      }

      let responseText = response.content || ''
      if (responseText.startsWith('with ')) {
        responseText = responseText.substring(5)
      }

      const newIds: Ids = {
        messageId: response.role + '_' + Date.now(),  // Generate a unique ID since response.id isn't available
        threadId: ids.threadId || `thread_${Date.now()}`
      }

      return [responseText, newIds]
    } catch (error) {
      if (error instanceof Error) {
        info(`Failed to send message to OpenAI: ${error.message}`)
        throw error
      }
      throw new Error('An unknown error occurred')
    }
  }
}
import type { ImageGenerationOptions } from 'internal:api'

export type SocketData = {
    namespace?: string | undefined
    event: string
    id?: string | undefined
    data: any
}

export type DenoiseLatentData = {
    step: number
    preview: Record<string, ImageGenerationOptions>
}

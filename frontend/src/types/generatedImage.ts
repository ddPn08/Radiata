export interface GeneratedImage {
    url: string
    info: Record<string, any>
}

export interface GeneratorImageProgress {
    type: string
    progress: number
    performance: number
}
export interface GeneratorImageResult {
    type: string
    info: { [key: string]: string | number | boolean }
    path: string[]
    perf: { [key: string]: number }
    performance: number
}

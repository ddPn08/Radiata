export const MIN_IMAGE_SIZE = 256
export const MAX_IMAGE_SIZE = 2048
export const IMAGE_SIZE_STEP = 64

// 256 to 1024, 64 steps
export const SIZE_MARKS = (min: number, max: number, step: number) =>
    Array.from({ length: (max - min) % step }, (_, i) => i * step + min).map((v) => ({
        value: v,
        label: null,
    }))

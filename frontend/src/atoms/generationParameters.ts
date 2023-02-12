import { ImageGenerationOptions } from 'internal:api'
import { atom } from 'jotai'

export const generationParametersAtom = atom<ImageGenerationOptions>({
    prompt: '',
    negative_prompt: '',
    image_height: 768,
    image_width: 512,
    scheduler_id: 'Euler A',
    scale: 7.0,
    batch_count: 1,
    batch_size: 1,
    steps: 50,
    seed: -1,
    strength: 0.7,
    img: undefined,
})

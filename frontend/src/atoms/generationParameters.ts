import { atom } from 'jotai'

import { SchedulerName } from '~/types/generate'

export interface GenerationParameters {
    prompt: string
    negative_prompt: string
    image_height: number
    image_width: number
    scheduler_id: string
    scale: number
    batch_count: number
    steps: number
    seed: number
    strength: number
    img: string | undefined
}

export interface GenerationParamertersForm extends GenerationParameters {
    scheduler_id: SchedulerName
}

// value for frontend
export const generationParametersAtom = atom<GenerationParamertersForm>({
    prompt: '',
    negative_prompt: '',
    image_height: 768,
    image_width: 512,
    scheduler_id: 'Euler A',
    scale: 7.0,
    batch_count: 1,
    steps: 50,
    seed: -1,
    strength: 0.7,
    img: undefined,
})

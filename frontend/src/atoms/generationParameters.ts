import { atom } from 'jotai'
import { SchedulerName } from '~/types/generate'

export interface GenerationParameters {
    prompt: string
    negative_prompt: string
    image_height: number
    image_width: number
    scheduler_id: SchedulerName
    scale: number
    batch_count: number
    steps: number
    seed: number
    strength: number
    img: string | undefined
}

export interface GenerationParamertersForm extends GenerationParameters {}

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
    strength: 1,
    img: undefined,
})

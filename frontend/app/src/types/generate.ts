export enum Scheduler {
    'DDIM' = 'ddim',
    'DEIS' = 'deis',
    'DPM2' = 'dpm2',
    'DPM2 A' = 'dpm2-a',
    'Euler' = 'euler',
    'Euler A' = 'euler_a',
    'Heun' = 'heun',
    'DPM++' = 'dpm++',
    'DPM' = 'dpm',
    'PNDM' = 'pndm',
}

export type SchedulerName = keyof typeof Scheduler
export const schedulerNames: SchedulerName[] = Object.keys(Scheduler) as SchedulerName[]

export const categoryList = ['txt2img', 'img2img'] as const
export type categoryType = (typeof categoryList)[number]

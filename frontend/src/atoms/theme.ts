import { ColorScheme } from '@mantine/core'
import { atomWithStorage } from 'jotai/utils'

export const themeAtom = atomWithStorage<ColorScheme | undefined>('theme', 'dark')

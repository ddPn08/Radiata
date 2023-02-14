import type { TablerIconsProps } from '@tabler/icons-react'
import type React from 'react'

export interface Tab {
    id: string
    label: string
    icon: (props: TablerIconsProps) => JSX.Element
    component: React.FC
}

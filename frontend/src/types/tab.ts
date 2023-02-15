import { TablerIconsProps } from '@tabler/icons-react'

export type TabType = 'generator' | 'engine' | 'imagesBrowser'

export interface Tab {
    id: TabType
    label: string
    icon: (props: TablerIconsProps) => JSX.Element
}

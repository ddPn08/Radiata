import { TablerIconsProps } from '@tabler/icons-react'

export interface Tab {
    id: string
    label: string
    icon: (props: TablerIconsProps) => JSX.Element
}

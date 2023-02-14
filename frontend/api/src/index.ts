import type React from 'react'

export type PluginMeta = {
    name: string
    main: string
    version: string
    author: string
    url: string
    frontend: {
        entryPoint: string
    }
}

export type Tab = {
    id: string
    label: string
    icon: React.FC
    component: React.FC
}

export * from './plugin'

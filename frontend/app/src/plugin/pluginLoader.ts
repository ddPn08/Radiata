import type { FrontEndPluginData } from '@lsmith/api'
import type { PluginMetaData } from 'internal:api'

import { api, createUrl } from '~/api'

export type PluginData = {
    meta: PluginMetaData
    data: FrontEndPluginData
}

export const plugins: PluginData[] = []

export const loadPlugins = async () => {
    const res = await api.pluginList()
    for (const plugin of res.data) {
        try {
            const m = await import(createUrl(`/api/plugins/js/${plugin.meta.name}`))
            plugins.push({
                meta: plugin.meta,
                data: m.default(),
            })
        } catch (error) {
            console.error(`Failed to load plugin: ${plugin.meta.name}`)
        }
    }
}

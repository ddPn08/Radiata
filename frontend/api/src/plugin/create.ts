import type { Tab } from '..'

export type FrontEndPluginData = {
    tabs: Tab[]
}
export type FrontEndPlugin = () => FrontEndPluginData | Promise<FrontEndPluginData>
export const createPlugin = (fn: FrontEndPlugin) => fn

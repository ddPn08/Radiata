/**
 * @typedef PluginMeta
 * @property {string} main
 * @property {string} name
 * @property {string} version
 * @property {string} author
 * @property {string} url
 * @property {string} javascript
 */

export const loadPlugins = async () => {
    /** @type {PluginMeta[]} */
    const plugins = await fetch('/api/plugins').then((res) => res.json())
    for (const dir in plugins) {
        const plugin = plugins[dir]
        if (!plugin.javascript) continue
        await import(`/api/plugins/${dir}/js/${plugin.javascript}`)
    }
}

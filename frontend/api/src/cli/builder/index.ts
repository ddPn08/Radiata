import esbuild from 'esbuild'
import fs from 'fs'
import path from 'path'
import type { PluginMeta } from 'src'

import { GlobalsResolver } from './plugins/global-resolver'

const loadConfig = (dir = process.cwd()): [PluginMeta, string] => {
    const filepath = path.join(dir, 'plugin.json')
    if (!fs.existsSync(filepath)) {
        const parent = path.dirname(dir)
        if (parent === dir) throw new Error('Could not find plugin meta file.')
        return loadConfig(parent)
    }
    const txt = fs.readFileSync(path.join(dir, 'plugin.json'), 'utf-8')
    return [JSON.parse(txt), dir]
}

export const run = async () => {
    const [config, dir] = loadConfig()
    const entry = path.isAbsolute(config.frontend.entryPoint)
        ? config.frontend.entryPoint
        : path.join(dir, config.frontend.entryPoint)

    const define: Record<string, any> = {}

    for (const k in process.env) {
        if (k.startsWith('REACT_APP_')) {
            define[`process.env.${k}`] = JSON.stringify(process.env[k])
        }
    }

    await esbuild.build({
        entryPoints: [entry],
        format: 'esm',
        outfile: path.join(dir, 'main.js'),
        jsxFactory: 'jsx',
        jsxFragment: 'Fragment',
        bundle: true,
        platform: 'browser',
        plugins: [GlobalsResolver],
        define,
    })
}

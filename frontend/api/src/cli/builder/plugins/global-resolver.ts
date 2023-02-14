import packageJson from '@lsmith/app/package.json'
import type { Plugin } from 'esbuild'

const PLUGIN_NAME = 'aviutl-toys:resolver'

const createRegExp = (globals: string[]) => {
  const raw: string[] = []
  for (const global of globals) {
    raw.push(`${global}(/.*)*`)
  }
  return new RegExp(raw.join('|'))
}

export const GlobalsResolver: Plugin = {
  name: PLUGIN_NAME,
  setup(build) {
    const deps = Object.keys(packageJson.dependencies)
    const filter = new RegExp(createRegExp(deps))
    build.onResolve({ filter }, (args) => {
      if (!deps.includes(args.path)) return
      const contents = `const p=window['__LSMITH_GLOBALS']['${args.path}'];module.exports=p`
      return {
        namespace: PLUGIN_NAME,
        path: args.path,
        pluginData: {
          contents,
        },
      }
    })
    build.onLoad({ filter, namespace: PLUGIN_NAME }, async (args) => {
      return {
        contents: args.pluginData.contents,
        loader: 'js',
      }
    })
  },
}

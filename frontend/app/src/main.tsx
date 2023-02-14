import React from 'react'
import ReactDOM from 'react-dom/client'

import App from './app'
import { loadPlugins } from './plugin/pluginLoader'

window['__LSMITH_GLOBALS'] = {}
/** @LOAD_GLOBALS */

await loadPlugins()

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

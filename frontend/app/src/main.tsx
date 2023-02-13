import React from 'react'
import ReactDOM from 'react-dom/client'

import App from './app'

window['__LSMITH_GLOBALS'] = {}
/** @LOAD_GLOBALS */

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)

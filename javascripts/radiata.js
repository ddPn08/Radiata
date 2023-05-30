import { gradioApp } from './utils'
import { loadPlugins } from './plugin'

window['__RADIATA_CONTEXT'] = {
    executedOnLoaded: false,
    eventListeners: {},
}

/**
 * @param {string} type
 * @param {*} listener
 * @param {*} options
 */
export const addEventListener = (type, listener, options) => {
    if (!window['__RADIATA_CONTEXT']['eventListeners'][type]) window['__RADIATA_CONTEXT']['eventListeners'][type] = []
    window['__RADIATA_CONTEXT']['eventListeners'][type].push({ listener, options })
}

/**
 * @param {string} type
 * @param  {...any} args
 * @returns
 */
export const callEventListeners = (type, ...args) => {
    if (!window['__RADIATA_CONTEXT']['eventListeners'][type]) return
    window['__RADIATA_CONTEXT']['eventListeners'][type].forEach((listener) => listener.listener(...args))
}

export const initialize = () => {
    loadPlugins()

    addEventListener('ready', () => {
        const button = gradioApp().getElementById('inference-mode-reload-button')
        button.click()
    })

    document.addEventListener('DOMContentLoaded', function () {
        var mutationObserver = new MutationObserver((m) => {
            if (window['__RADIATA_CONTEXT'].executedOnLoaded) return
            window['__RADIATA_CONTEXT'].executedOnLoaded = true
            const interval = setInterval(() => {
                const root = gradioApp().getElementById('radiata-root')
                if (root) {
                    clearInterval(interval)
                    callEventListeners('ready')
                }
            }, 500)
        })
        mutationObserver.observe(gradioApp(), { childList: true, subtree: true })
    })
}

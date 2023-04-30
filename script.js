window['__RADIATA_CONTEXT'] = {
    executedOnLoaded: false
}

/**
 * @returns {ShadowRoot | Document}
 */
function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app')
    const elem = elems.length == 0 ? document : elems[0]

    if (elem !== document) elem.getElementById = (id) => document.getElementById(id)
    return elem.shadowRoot ? elem.shadowRoot : elem
}

document.addEventListener("DOMContentLoaded", function () {
    var mutationObserver = new MutationObserver((m) => {
        if (window['__RADIATA_CONTEXT'].executedOnLoaded) return
        window['__RADIATA_CONTEXT'].executedOnLoaded = true
        const interval = setInterval(() => {
            const root = gradioApp().getElementById("radiata-root")
            if (root) {
                clearInterval(interval)
                const button = gradioApp().getElementById("inference-mode-reload-button")
                button.click()
            }
        }, 500)
    })
    mutationObserver.observe(gradioApp(), { childList: true, subtree: true })
})
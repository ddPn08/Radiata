/**
 * Get the gradio-app element
 * @returns {ShadowRoot | Document}
 */
export const gradioApp = () => {
    const elems = document.getElementsByTagName('gradio-app')
    const elem = elems.length == 0 ? document : elems[0]

    if (elem !== document) elem.getElementById = (id) => document.getElementById(id)
    return elem.shadowRoot ? elem.shadowRoot : elem
}

/**
 * Set a MutationObserver on a target
 * @param {Element} target
 * @param {*} fn
 * @returns {MutationObserver}
 */
export const setObserver = (target, fn) => {
    let observer = new MutationObserver((records) => {
        for (const record of records) for (const nodeList of record.addedNodes) if (nodeList instanceof HTMLElement) fn(nodeList)
    })
    observer.observe(target, { childList: true, subtree: true })
    return observer
}

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

function setObserver(target, fn) {
    let observer = new MutationObserver((records) => {
        for (const record of records)
            for (const nodeList of record.addedNodes)
                if (nodeList instanceof HTMLElement) fn(nodeList)
    });
    observer.observe(target, { childList: true, subtree: true })
    return observer
}

document.addEventListener("DOMContentLoaded", function () {
    var mutationObserver = new MutationObserver((m) => {
        if (window['__RADIATA_CONTEXT'].executedOnLoaded) return
        window['__RADIATA_CONTEXT'].executedOnLoaded = true
        const interval = setInterval(() => {
            const root = gradioApp().getElementById("radiata-root")
            if (root) {
                clearInterval(interval)
                inferenceReloadListeners()
                attachGalleryListeners()
            }
        }, 500)
    })
    mutationObserver.observe(gradioApp(), { childList: true, subtree: true })
})

function inferenceReloadListeners() {
    const button = gradioApp().getElementById("inference-mode-reload-button")
    button.click()
}

/* Gallery */

function attachGalleryListeners() {
    gradioApp().querySelectorAll('.info-gallery').forEach((gallery) => {
        setObserver(gallery, (node) => {
            node.querySelector(":scope>button[aria-label=Clear]")?.addEventListener('click', click);
        });
        function click() {
            let textarea = gallery.parentElement.querySelector(".image-generation-selected textarea");
            let selected = [...gallery.querySelectorAll('.thumbnail-item.thumbnail-small')].find((e) => e.classList.contains("selected"));
            let src = selected?.getElementsByTagName("img")[0].getAttribute("src")
            textarea.value = src == null ? "" : new URL(src).pathname.slice(6);
            textarea.dispatchEvent(new Event('input'));
        }
        gallery.addEventListener('click', click);
    });
}

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
    function setObserver(gallery) {
        new MutationObserver((records) => {
            for (const record of records) {
                for (const nodeList of record.addedNodes) {
                    if (nodeList instanceof HTMLElement) setEventHandler(gallery, nodeList)
                }
            }
        }).observe(gallery, { childList: true, subtree: true })
    }
    function setEventHandler(gallery, node) {
        console.log(node.innerText);
        console.log(node);
        if (node.tagName == "BUTTON" || node.classList.contains("preview")) {
            node.addEventListener('click', () => {
                gradioApp().getElementById(gallery.id + "-button").click();
            });
        } else if (node.innerText.includes("processing")) {
            console.log(node)
            gradioApp().getElementById(gallery.id + "-button").click();
        } else {
            node.querySelector("button[aria-label=Clear]")?.addEventListener('click', () => {
                gradioApp().getElementById(gallery.id + "-button").click();
            });
        }
    }
    gradioApp().querySelectorAll('.info-gallery').forEach((gallery) => setObserver(gallery, true));
}


function idEscape(id) {
    return id.replace(/(:|\.|\[|\]|,|=|@|\s)/g, "\\$1");
}
function selectedGalleryButton(id) {
    return [...gradioApp().querySelectorAll('#' + idEscape(id) + ' .thumbnail-item.thumbnail-small')].findIndex((e) => e.classList.contains("selected"));
}

function selectedTab(id) { // https://github.com/gradio-app/gradio/issues/3793
    return [...gradioApp().querySelectorAll('#' + idEscape(id) + ' button')].find((e) => e.classList.contains("selected")).innerText;
}

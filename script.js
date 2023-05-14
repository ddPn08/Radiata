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
    function setEventHandler(gallery, recursive = false) {
        gallery.querySelectorAll('button, .preview').forEach((image, key) => {
            image.addEventListener('click', () => {
                if (recursive) setEventHandler(gallery);
                gradioApp().getElementById(gallery.id + "-button").click();
            });
        });
    }
    let galleryList = gradioApp().querySelectorAll('.info-gallery');
    galleryList.forEach((gallery) => setEventHandler(gallery, true));
}


function selectedGalleryButton(id) {
    return [...gradioApp().querySelectorAll('#' + id + ' .thumbnail-item.thumbnail-small')].findIndex((e) => e.classList.contains("selected"));
}

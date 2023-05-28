import { addEventListener } from './radiata'
import { gradioApp, setObserver } from './utils'

addEventListener('ready', () => {
    gradioApp()
        .querySelectorAll('.info-gallery')
        .forEach((gallery) => {
            const click = () => {
                let textarea = gallery.parentElement.querySelector('.image-generation-selected textarea')
                let selected = [...gallery.querySelectorAll('.thumbnail-item.thumbnail-small')].find((e) => e.classList.contains('selected'))
                let src = selected?.getElementsByTagName('img')[0].getAttribute('src')
                textarea.value = src == null ? '' : new URL(src).pathname.slice(6)
                textarea.dispatchEvent(new Event('input'))
            }
            setObserver(gallery, (node) => {
                node.querySelector(':scope>button[aria-label=Clear]')?.addEventListener('click', click)
            })
            gallery.addEventListener('click', click)
        })
})

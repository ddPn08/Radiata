import { addEventListener } from './radiata'
import { gradioApp } from './utils'

class InputTypingWord {

    splitList = [",", " ", "\n"]

    update(text) {
        this.value = this.wordSplit(text);
    }

    /* Get the word being typed from the position of the input bar */
    getTypingWord = (selection) => {
        return this.value[this.getTypingIndex(selection)];
    }

    getTypingIndex = (selection) => {
        return this.value.findIndex((e, i) => this.value.slice(0, i + 1).reduce((v, e) => v + e.length, 0) > selection - 1);
    }

    /* Separate by word, (e.g. '1,2,3' => ['1', ',', '2', ',', '3']) */
    wordSplit = (text) => {
        let value = [text];
        this.splitList.forEach((s) => value = value.map((e) => e.split(s).length == 1 ? [e] : e.split(s).map((f) => [f, s]).flat()).flat());
        value.length > 1 && value.pop();
        return value;
    }

    lineSplit = (text) => {
        return text.split("\n");
    }
}
class AutoComplete {
    max = 20;
    data = ["apple", "application", "app", "banana", "cherry", "grape", "lemon", "lime", "mango", "melon", "orange", "peach", "pear", "pineapple", "plum", "strawberry", "watermelon"];
    getAutoComplete = (text) => {
        const select = this.data.filter((e) => e.slice(0, text.length) === text);
        return select.sort((a, b) => a.length - b.length).slice(0, this.max)
    }

}


addEventListener('ready', () => {
    const autoAutoComplete = new AutoComplete();
    [...gradioApp().querySelectorAll('textarea[placeholder="Prompt"]')].forEach((area) => {

        const ul = document.createElement('ul');
        ul.classList.add("autocomplete");
        ul.classList.add("none");

        const areaInputValue = new InputTypingWord(area);

        document.addEventListener('keydown', (e) => {
            if (document.activeElement.parentElement != ul) return;
            if ([" ", "Enter"].includes(e.key)) {
                select(document.activeElement.textContent);
                area.focus();
            } else if (["Tab", "ArrowDown"].includes(e.key)) {
                const next = document.activeElement.nextElementSibling;
                if (next) next.focus();
                else document.activeElement.parentElement.firstElementChild.focus()
                e.preventDefault();
            } else if (e.key == "ArrowUp") {
                const prev = document.activeElement.previousElementSibling;
                if (prev) prev.focus();
                else document.activeElement.parentElement.lastElementChild.focus()
                e.preventDefault();
            } else if (e.key == "Escape") {
                ul.classList.add("none");
                e.preventDefault();
            } else {
                area.focus();
            }
        });


        document.addEventListener('click', (e) => {
            if (e.target.parentElement == ul) select(e.target.textContent);
            ul.classList.add("none");
        });

        const select = (text) => {
            if (area.selectionStart != area.selectionEnd) return;
            const wordIndex = areaInputValue.getTypingIndex(area.selectionStart);
            areaInputValue.value[wordIndex] = text;
            area.value = areaInputValue.value.join("");
            ul.classList.add("none");
        }

        const event = (e) => {
            areaInputValue.update(area.value);

            const lineList = areaInputValue.lineSplit(e.target.value);
            ul.style.top = (lineList.length + 1) * 1.5 + "em";
            const word = areaInputValue.getTypingWord(area.selectionStart);
            while (ul.firstChild) ul.removeChild(ul.firstChild);

            if (word.length < 1) ul.classList.add("none");
            else {
                const complete = autoAutoComplete.getAutoComplete(word);
                if (complete.length > 0) ul.classList.remove("none")
                else ul.classList.add("none");
                complete.forEach((e) => {
                    const li = document.createElement('li');
                    li.setAttribute("tabindex", "0");
                    li.textContent = e;
                    ul.appendChild(li);
                });
            }
        }

        area.after(ul);
        area.addEventListener('keyup', event);
        area.addEventListener('keydown', event);
    });

});

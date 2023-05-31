import { addEventListener } from '/javascripts/radiata'
import { gradioApp } from '/javascripts/utils'

import { parse } from 'https://cdn.jsdelivr.net/npm/csv-parse@5.4.0/+esm'


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
    data = [];


    // https://github.com/DominikDoom/a1111-sd-webui-tagcomplete/blob/main/LICENSE
    version = "2.5.0";
    url = `https://cdn.jsdelivr.net/gh/DominikDoom/a1111-sd-webui-tagcomplete@${this.version}/tags/danbooru.csv`
    color = {
        "-1": ["red", "maroon"],
        "0": ["lightblue", "dodgerblue"],
        "1": ["indianred", "firebrick"],
        "3": ["violet", "darkorchid"],
        "4": ["lightgreen", "darkgreen"],
        "5": ["orange", "darkorange"]
    }


    read = () => new Promise(async (resolve, reject) => {
        const parser = parse();
        parser.on('readable', () => {
            let record;
            while (record = parser.read()) {
                const [value, tag, number, alias] = record;
                this.data.push([value, tag, Number(number), alias.split(",")]);
            }
        });
        parser.on('error', (err) => reject(err));
        parser.on('end', () => resolve());
        const res = await fetch(this.url);
        parser.write(await res.text());
        parser.end();
    });

    toColor = (value, darkTheme) => {
        return this.color[value][darkTheme ? 1 : 0];
    }

    toCompacting = (number) => {
        if (number < 1000) return number;
        else if (number < 1000000) return (number / 1000).toFixed(1) + "K";
        else return (number / 1000000).toFixed(1) + "M";
    }


    getAutoComplete = (text) => {
        const [selectValue, selectAlias] = [[], []];
        this.data.forEach(([value, tag, number, alias]) => {
            if (value.includes(text)) return selectValue.push([value, tag, number, alias]);
            const aliasFind = alias.find((e) => e.includes(text));
            if (aliasFind) selectAlias.push([value, tag, number, alias, aliasFind]);
        });
        const select = [...selectValue, ...selectAlias];
        return select.sort(([, , a], [, , b]) => b - a).slice(0, this.max)
    }
}


addEventListener('ready', async () => {
    const autoComplete = new AutoComplete();
    await autoComplete.read();
    const darkTheme = document.querySelector('body').classList.contains('dark');
    [...gradioApp().querySelectorAll('textarea[placeholder="Prompt"]')].forEach((area) => {

        const ul = document.createElement('ul');
        ul.classList.add("autocomplete");
        ul.classList.add("none");
        ul.setAttribute("tabindex", "-1");

        const areaInputValue = new InputTypingWord(area);

        document.addEventListener('keydown', (e) => {
            if (document.activeElement == area) {
                if (["Tab", "ArrowDown"].includes(e.key)) {
                    ul.firstElementChild.focus();
                    e.preventDefault();
                    return;
                }
            }
            if (document.activeElement.parentElement != ul) return;
            if ([" ", "Enter"].includes(e.key)) {
                select(document.activeElement.getElementsByTagName("span")[1].innerText);
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
            const parent = e.target.tagName == "ul" ? e.target : e.target.closest("ul");
            if (ul != parent) ul.classList.add("none");
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
                const complete = autoComplete.getAutoComplete(word);
                if (complete.length > 0) ul.classList.remove("none")
                else ul.classList.add("none");
                complete.forEach(([value, tag, number, alias, aliasFind]) => {

                    const li = document.createElement('li');
                    li.setAttribute("tabindex", "0");
                    ul.appendChild(li);

                    const spanBefor = document.createElement('span');
                    spanBefor.style.color = autoComplete.toColor(tag, darkTheme);
                    if (aliasFind) spanBefor.textContent = aliasFind + "⇢";
                    li.appendChild(spanBefor);


                    const spanCenter = document.createElement('span');
                    spanCenter.style.color = autoComplete.toColor(tag, darkTheme);
                    spanCenter.textContent = value;
                    li.appendChild(spanCenter);

                    const spanAfter = document.createElement('span');
                    spanAfter.textContent = autoComplete.toCompacting(number);
                    li.appendChild(spanAfter);

                    li.addEventListener('click', () => select(value));
                });
            }
        }

        area.after(ul);
        area.addEventListener('keyup', event);
        area.addEventListener('keydown', event);
    });

});

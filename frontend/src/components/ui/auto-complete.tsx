import { styled } from 'decorock'
import {
  Component,
  createEffect,
  createReaction,
  createSignal,
  For,
  onCleanup,
  onMount,
  Show,
} from 'solid-js'
import { isServer } from 'solid-js/web'

import { Input } from './input'

type PropsT = {
  suggestions: ((value: string) => Promise<string[]> | string[]) | string[]
  onInput?: (value: string) => void
  onChange?: (value: string) => void
  value: string
  limit?: number
  confirmKey?: string[]

  placeholder?: string
  class?: string
  error?: string | undefined
}

const Container = styled.div`
  position: relative;
`

const Suggestions = styled.div`
  position: absolute;
  z-index: 5;
  top: 100%;
  left: 0;
  display: flex;
  overflow: hidden;
  width: 100%;
  max-height: 330px;
  flex-direction: column;
  padding: 0.5rem;
  border-radius: 0.25rem;
  margin-top: 0.5rem;
  background-color: ${(p) => p.theme.colors.secondary};
  box-shadow: 0 0 16px -6px rgba(0, 0, 0, 0.6);
  overflow-y: auto;
`

const Item = styled.div<{ selecting: boolean }>`
  width: 100%;
  padding: 0.5rem;
  background-color: ${(p) =>
    p.selecting ? p.theme.colors.secondary.darken(0.25) : p.theme.colors.secondary};
  cursor: pointer;
  text-align: left;
  transition: 0.25s;

  &:hover {
    background-color: ${(p) => p.theme.colors.secondary.darken(0.25)};
  }
`

export const AutoComplete: Component<PropsT> = (props) => {
  const [suggestions, setSuggestions] = createSignal<string[]>([])
  const [currentIndex, setCurrentIndex] = createSignal(-1)
  const [inputting, setInputting] = createSignal(false)
  const [task, setTask] = createSignal(0)

  const track = createReaction(() => {
    track(() => props.value)
    clearInterval(task())
    setTask(0)
  })
  track(() => props.value)

  createEffect(() => {
    if (Array.isArray(props.suggestions)) {
      setSuggestions(
        props.suggestions.filter(
          (v) => !props.value || v.match(new RegExp('^' + props.value, 'i')),
        ),
      )
    } else {
      if (task() === 0)
        setTask(
          setTimeout(async () => {
            if (typeof props.suggestions === 'function')
              setSuggestions(await props.suggestions(props.value))
          }, 500) as any,
        )
    }
  })

  let ref: HTMLDivElement

  const listener = (e: MouseEvent) => {
    const isThis = ref === e.target || ref.contains(e.target as Node)
    if (inputting() && !isThis) setInputting(false)
  }
  onMount(() => {
    if (!isServer) window.addEventListener('click', listener)
  })
  onCleanup(() => {
    if (!isServer) window.removeEventListener('click', listener)
  })

  return (
    <Container
      ref={ref!}
      class={props.class}
      onKeyDown={(e) => {
        if (e.key === 'Tab') {
          if (e.isComposing) return
          e.preventDefault()
          if (!inputting()) setInputting(true)
          if (currentIndex() >= suggestions().length) setCurrentIndex(-1)
          else setCurrentIndex(currentIndex() + 1)
        }
        if (props.confirmKey?.includes(e.key) || e.key === 'Enter') {
          if (currentIndex() === -1) {
            props.onChange?.(props.value)
            return
          }
          e.preventDefault()
          props.onInput?.(suggestions()[currentIndex()] || '')
          setCurrentIndex(-1)
          setInputting(false)
        }
      }}
    >
      <Input
        value={props.value}
        placeholder={props.placeholder || ''}
        onInput={(e) => {
          setInputting(true)
          // eslint-disable-next-line no-irregular-whitespace
          props.onInput?.(e.currentTarget.value.trim().replace(/　/g, ''))
        }}
        onFocusIn={() => setInputting(true)}
        error={props.error}
      />
      <Show when={inputting() && suggestions().length > 0}>
        <Suggestions>
          <For each={suggestions().slice(0, props.limit || 10)}>
            {(value, i) => (
              <Item
                selecting={i() === currentIndex()}
                onClick={() => {
                  setInputting(false)
                  props.onInput?.(value)
                  props.onChange?.(value)
                }}
              >
                {value}
              </Item>
            )}
          </For>
        </Suggestions>
      </Show>
    </Container>
  )
}

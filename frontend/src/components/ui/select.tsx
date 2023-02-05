import { css, styled, useTheme } from 'decorock'
import { Component, createMemo, createSignal, For, onCleanup, onMount, Show } from 'solid-js'
import { isServer } from 'solid-js/web'

import IconExpandMore from '~icons/material-symbols/expand-more'

type Option = {
  value: string
  label: string
}

const Container = styled.div`
  position: relative;
  font-size: 0.8rem;
  user-select: none;
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
  background-color: ${(p) => p.theme.colors.secondary};
  cursor: pointer;
  text-align: left;
  transition: 0.25s;

  &:hover {
    background-color: ${(p) => p.theme.colors.secondary.darken(0.25)};
  }
`

export const Select: Component<{
  options: Option[]
  value: string
  onChange?: (option: Option) => void
}> = (props) => {
  const theme = useTheme()
  const [ref, setRef] = createSignal<HTMLDivElement>()
  const [selecting, setSelecting] = createSignal(false)
  const listener = (e: MouseEvent) => {
    const el = ref()!
    const isThis = el === e.target || el.contains(e.target as Node)
    if (selecting() && !isThis) setSelecting(false)
  }
  onMount(() => {
    if (!isServer) window.addEventListener('click', listener)
  })
  onCleanup(() => {
    if (!isServer) window.removeEventListener('click', listener)
  })
  const current = createMemo(() => props.options.find((v) => v.value === props.value))

  return (
    <Container ref={setRef}>
      <div
        class={css`
          display: inline-flex;
          width: 100%;
          box-sizing: border-box;
          align-items: center;
          justify-content: space-between;
          padding: 0.5rem;
          border: 1px solid ${theme.colors.primary.fade(0.5).string()};
          border-radius: 0.5rem;
          background-color: rgba(0, 0, 0, 0.05);
          color: ${theme.colors.primary};
          cursor: pointer;

          svg {
            font-size: 1rem;
          }
        `}
        onClick={() => setSelecting(!selecting())}
      >
        <div>{current()?.label}</div>
        <IconExpandMore />
      </div>
      <Show when={selecting()}>
        <Suggestions>
          <For each={props.options}>
            {(value, i) => (
              <Item
                selecting={i() === props.options.findIndex((v) => v.value === props.value)}
                onClick={() => {
                  setSelecting(false)
                  props.onChange?.(value)
                }}
              >
                {value.label}
              </Item>
            )}
          </For>
        </Suggestions>
      </Show>
    </Container>
  )
}

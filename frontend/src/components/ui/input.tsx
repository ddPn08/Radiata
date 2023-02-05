import Color from 'color'
import { css, styled } from 'decorock'
import { Component, ComponentProps, createEffect, createSignal, Show, splitProps } from 'solid-js'

import { Required } from './required'

const Container = styled.div`
  text-align: left;
`

const StyledInput = styled.input`
  display: inline-block;
  width: 100%;
  box-sizing: border-box;
  padding: 0.5rem;
  border: 1px solid ${(p) => p.theme.colors.primary.fade(0.5).string()};
  border-radius: 0.5rem;
  background-color: rgba(0, 0, 0, 0.05);
  color: ${(p) => (p.disabled ? p.theme.colors.primary.fade(0.25) : p.theme.colors.primary)};
  font-size: 1rem;
  outline: none;

  &:focus {
    border: 1px solid ${(p) => p.theme.colors.primary.darken(0.25).string()};
  }
`
export const Input: Component<
  ComponentProps<'input'> & {
    error?: boolean | string | undefined
  }
> = (props) => {
  const [local, others] = splitProps(props, ['class', 'error', 'onInput'])
  const [changed, setChanged] = createSignal(false)
  createEffect(() => {
    if (props.error) setChanged(false)
  })
  return (
    <Container class={local.class}>
      <Show when={props.required}>
        <Required>必須</Required>
      </Show>
      <StyledInput
        class={
          local.error && !changed()
            ? css`
                border: 1px solid ${Color('red').lighten(0.25).string()};
              `
            : ''
        }
        onInput={(e) => {
          setChanged(true)
          if (typeof local.onInput === 'function') local.onInput(e)
        }}
        {...others}
      />
      <Show when={local.error && !changed()}>
        <div
          class={css`
            color: ${Color('red').lighten(0.25).string()};
          `}
        >
          {typeof local.error === 'string' ? local.error : ''}
        </div>
      </Show>
    </Container>
  )
}

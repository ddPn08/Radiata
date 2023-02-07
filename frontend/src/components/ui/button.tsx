import { css, styled } from 'decorock'
import {
  Component,
  ComponentProps,
  createMemo,
  createSignal,
  For,
  Show,
  splitProps,
} from 'solid-js'

import { CircleSpinner } from './spinner'

const StyledButton = styled.button<{ _padding: boolean }>`
  position: relative;
  ${(p) => (p._padding ? '' : 'padding: 0.75rem 1.5rem;')}

  border: none;
  border-radius: 1rem;
  margin: 0.5rem;
  background-color: ${(p) => p.theme.colors.primary.fade(0.8)};
  color: ${(p) => p.theme.colors.primary};
  cursor: pointer;
  font-size: medium;
  font-weight: bold;
  outline: none;
  text-align: center;
  text-decoration: none;
  transition: 0.2s;

  &:disabled {
    pointer-events: none;
  }

  &:hover {
    background-color: ${(p) => p.theme.colors.primary.fade(0.7)};
  }

  &:active {
    div {
      color: ${(p) => p.theme.colors.primary.lighten(0.7)};
    }

    background-color: ${(p) => p.theme.colors.primary.fade(0.85)};
  }
`

const Inner = styled.div<{ disabled: boolean }>`
  width: 100%;
  height: 100%;
  color: ${(p) => p.theme.colors.primary.fade(p.disabled ? 0.5 : 0)};
`

const Status = styled.div`
  margin-top: 0.5rem;
  color: ${(p) => p.theme.colors.primary.fade(0.5)};
  font-size: 0.9rem;
`

export const Button: Component<
  ComponentProps<'button'> & {
    task?: (
      e: MouseEvent & {
        currentTarget: HTMLButtonElement
        target: Element
      },
    ) => any
    loading?: boolean
    status?: string | undefined
    _padding?: boolean | undefined
  }
> = (props) => {
  const [local, others] = splitProps(props, ['children', 'loading', 'status', 'onClick', 'task'])
  const [running, setRunning] = createSignal(false)
  const loading = createMemo(() => local.loading || running())
  return (
    <StyledButton
      {...others}
      _padding={!!props._padding}
      onClick={(e) => {
        if (loading()) return e.preventDefault()
        if (typeof local.task === 'function') {
          setRunning(true)
          const t = local.task(e)
          if (typeof t !== 'function' && t.then) {
            t.then(() => setRunning(false))
            t.catch(() => setRunning(false))
          }
        }
        if (typeof local.onClick === 'function') local.onClick(e)
      }}
    >
      <Inner
        class={css`
          display: flex;
          align-items: center;
          justify-content: center;
        `}
        disabled={!!props.disabled}
      >
        <Show when={!loading()} fallback={<CircleSpinner />}>
          {local.children}
        </Show>
      </Inner>
      <Show when={loading() && local.status}>
        <Status>
          <For each={props.status?.split('\n')}>
            {(line, i) => (
              <>
                <Show when={i() !== 0}>
                  <br />
                </Show>
                {line}
              </>
            )}
          </For>
        </Status>
      </Show>
    </StyledButton>
  )
}

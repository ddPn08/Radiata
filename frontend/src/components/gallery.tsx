import { css, styled, useTheme } from 'decorock'
import { Component, createEffect, createSignal, For, on, Show } from 'solid-js'

import { createUrl } from '~/api'

const Container = styled.div`
  position: relative;
  overflow: hidden;
  width: 100%;
  border-radius: 1rem;
`

const Inner = styled.div`
  display: flex;
  width: 100%;
  flex-wrap: wrap;
  align-content: flex-start;
  padding: 1rem;
  aspect-ratio: 1/1;
  background-color: ${(p) => p.theme.colors.secondary.darken(0.5)};
  gap: 1rem;
  overflow-y: auto;
`

const Full = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  display: flex;
  width: 100%;
  height: 100%;
  align-items: center;
  justify-content: center;
  background-color: ${(p) => p.theme.colors.secondary.darken(0.5)};

  img {
    object-fit: contain;
  }
`

const StyledItem = styled.div`
  display: inline-block;
  overflow: hidden;
  width: 200px;
  height: 200px;
  border-radius: 1rem;
  cursor: pointer;

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  &:hover {
    outline: solid 1px ${(p) => p.theme.colors.primary};
  }
`

const ImageItem: Component<{ src: string; info: Record<string, string>; onClick: () => void }> = (
  props,
) => {
  return (
    <StyledItem onClick={props.onClick}>
      <img src={props.src} alt="" />
    </StyledItem>
  )
}

export const Gallery: Component<{
  images: [string, Record<string, string>][]
  category: 'txt2img' | 'img2img'
}> = (props) => {
  const theme = useTheme()
  const [selected, setSelected] = createSignal<[string, Record<string, string>] | null>(null)

  createEffect(
    on(
      () => props.images,
      () => setSelected(null),
    ),
  )

  return (
    <Container>
      <Inner>
        <For each={props.images}>
          {([src, info]) => (
            <ImageItem
              src={createUrl(`/api/images/${props.category}/${src}`)}
              info={info}
              onClick={() => setSelected([src, info])}
            />
          )}
        </For>
      </Inner>
      <Show when={selected()} keyed>
        {([src]) => (
          <Full>
            <div
              class={css`
                width: 100%;
                height: 100%;
                padding: 1rem;

                img {
                  width: 100%;
                  height: 100%;

                  &:hover {
                    outline: solid 1px ${theme.colors.primary};
                  }
                }
              `}
            >
              <img
                src={createUrl(`/api/images/${props.category}/${src}`)}
                onClick={() => setSelected(null)}
              />
            </div>
          </Full>
        )}
      </Show>
    </Container>
  )
}

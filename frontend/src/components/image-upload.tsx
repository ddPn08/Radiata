import { createDropzone, createFileUploader, UploadFile } from '@solid-primitives/upload'
import clsx from 'clsx'
import { css, useTheme } from 'decorock'
import { ComponentProps, createEffect, createSignal, For, Show, splitProps } from 'solid-js'
import type { Component } from 'solid-js'

import { Button } from './ui/button'

import IconAdd from '~icons/material-symbols/add'
import IconClose from '~icons/material-symbols/close'
import IconFile from '~icons/material-symbols/file-copy'

const ImageBox: Component<
  ComponentProps<'div'> & {
    image: UploadFile
    onDelete: () => void
    selected: boolean
    onSelect: () => void
    removable: boolean
  }
> = (props) => {
  const theme = useTheme()
  const [local, others] = splitProps(props, [
    'image',
    'onDelete',
    'selected',
    'onSelect',
    'class',
    'onClick',
  ])
  return (
    <div
      class={clsx(
        local.class,
        css`
          position: relative;
          max-width: 100%;
          border: ${local.selected ? `3px solid ${theme.colors.secondary.darken(0.25)}` : 'none'};
          aspect-ratio: 1 / 1;
          background-color: gray;
          background-image: url(${local.image.source});
          background-position: 50%;
          background-repeat: no-repeat;
          background-size: contain;
        `,
      )}
      onClick={() => local.onSelect()}
      {...others}
    >
      <Show when={props.removable}>
        <Button
          onClick={() => local.onDelete()}
          class={css`
            position: absolute;
            left: 0;
            padding: 0.1rem;
            border-radius: 0;
          `}
        >
          <IconClose />
        </Button>
      </Show>
    </div>
  )
}

export const ImageUpload: Component<
  Omit<ComponentProps<'div'>, 'onChange' | 'onSelect'> & {
    onChange: (files: UploadFile[]) => Promise<void> | void
    images: UploadFile[]
    multiple?: boolean
    editable?: boolean
  } & (
      | { selectable?: false | undefined }
      | { selectable: true; selected: number; onSelect: (i: number) => void }
    )
> = (props) => {
  const theme = useTheme()
  const [images, setImages] = createSignal<UploadFile[]>([])

  createEffect(() => {
    setImages(props.images)
  })

  const onDrop = async (files: UploadFile[]) => {
    if (!files) return
    const all = props.multiple ? [...images(), ...files] : [...files]
    props.onChange(all)
    setImages(props.images)
    if (props.selectable) props.onSelect(all.length - 1)
  }

  const { selectFiles } = createFileUploader({ accept: 'image/*', multiple: !!props.multiple })
  const { setRef: dropzoneRef } = createDropzone({ onDrop })

  return (
    <div
      class={clsx(
        props.class,
        css`
          display: flex;
          width: 100%;
          flex-direction: column;
          align-items: center;
          padding: 0;
          gap: 1rem;
        `,
      )}
    >
      <div
        ref={dropzoneRef}
        onClick={() => {
          if (images().length > 0) return
          selectFiles(onDrop)
        }}
        class={css`
          width: 100%;
          padding: 5rem 0;
          background-color: ${theme.colors.primary.fade(0.9)};
          color: ${theme.colors.primary.fade(0.5)};
          cursor: ${images().length > 0 ? 'auto' : 'pointer'};
          text-align: center;
        `}
      >
        <Show
          when={images().length > 0}
          fallback={
            <>
              <IconFile height={75} width={75} />
              <p>Drop image here</p>
            </>
          }
        >
          <div
            class={css`
              display: flex;
              width: 100%;
              flex-wrap: wrap;
              align-items: center;
              justify-content: center;
              gap: 2rem;
            `}
          >
            <For each={images()}>
              {(image, i) => (
                <ImageBox
                  image={image}
                  removable={!!props.editable}
                  selected={!!props.selectable && props.selected === i()}
                  onSelect={() => props.selectable && props.onSelect(i())}
                  onDelete={() => {
                    const filtered = images().filter((p) => p.source !== image.source)
                    setImages(filtered)
                    props.onChange(filtered)
                  }}
                  class={css`
                    width: ${images().length === 1 ? '600px' : '150px'};
                    height: ${images().length === 1 ? '600px' : '150px'};
                  `}
                />
              )}
            </For>
            <Show when={props.multiple && props.editable}>
              <Button
                class={css`
                  svg {
                    width: 8rem;
                    height: 8rem;
                  }
                `}
                onClick={() => selectFiles(onDrop)}
              >
                <IconAdd />
              </Button>
            </Show>
          </div>
        </Show>
      </div>
    </div>
  )
}

import clsx from 'clsx'
import { css } from 'decorock'
import { Component, ComponentProps, splitProps } from 'solid-js'

import { Button } from './button'

export const IconButton: Component<ComponentProps<typeof Button>> = (props) => {
  const [local, others] = splitProps(props, ['class'])
  return (
    <Button
      {...others}
      _padding
      class={clsx(
        local.class,
        css`
          padding: 0.5rem;
          border-radius: 0.5rem;
          aspect-ratio: 1/1;

          svg {
            display: block;
          }
        `,
      )}
    />
  )
}

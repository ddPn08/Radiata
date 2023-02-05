import { styled } from 'decorock'
import type { ComponentProps } from 'solid-js'

type Props = ComponentProps<'div'> & {
  inline?: boolean | undefined
  gap?: string | number | undefined
}

export const HStack = styled.div<Props>`
  display: ${(p) => (p.inline ? 'inline-flex' : 'flex')};
  gap: ${(p) => p.gap || '0.5rem'};
`

export const VStack = styled.div<Props>`
  display: ${(p) => (p.inline ? 'inline-flex' : 'flex')};
  flex-direction: column;
  gap: ${(p) => p.gap || '0.5rem'};
`

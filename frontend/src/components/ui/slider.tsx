import { css, styled } from 'decorock'
import type { Component, ComponentProps } from 'solid-js'

import { Input } from './input'
import { HStack } from './stack'

const StyledInput = styled.input`
  width: 100%;
  height: 5px;
  border-radius: 6px;
  appearance: none;
  background-color: ${(e) => e.theme.colors.primary};

  &:focus,
  &:active {
    outline: none;
  }

  &::-webkit-slider-thumb {
    position: relative;
    display: block;
    width: 22px;
    height: 22px;
    border: 2px solid ${(e) => e.theme.colors.secondary.fade(0.2)};
    border-radius: 50%;
    appearance: none;
    background-color: #fff;
    cursor: pointer;
  }

  &:active::-webkit-slider-thumb {
    box-shadow: 0 0 0 4px ${(e) => e.theme.colors.primary.fade(0.1)};
    transition: 0.4s;
  }
`

export const Slider: Component<Omit<ComponentProps<'input'>, 'type'>> = (props) => {
  return <StyledInput type="range" {...props} />
}

export const WithSlider: Component<{
  label: string
  value: string | number
  onChange: (v: string) => void
  max: number
  min: number
  step: number
}> = (props) => {
  return (
    <div>
      <HStack
        class={css`
          width: 100%;
          justify-content: space-between;
        `}
      >
        <div>{props.label}</div>
        <Input value={props.value} onChange={(e) => props.onChange(e.currentTarget.value)} />
      </HStack>
      <Slider
        max={props.max}
        min={props.min}
        step={props.step}
        value={props.value}
        onInput={(e) => props.onChange(e.currentTarget.value)}
      />
    </div>
  )
}

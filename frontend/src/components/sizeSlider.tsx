import { Slider, SliderProps } from '@mantine/core'
import { Component, forwardRef, Ref, RefAttributes } from 'react'

// 256 to 1024, 64 steps
export const SIZE_MARKS = Array.from({ length: 16 }, (_, i) => i * 64 + 256).map((v) => ({
  value: v,
  label: null,
}))

type Props = SliderProps & RefAttributes<HTMLDivElement>

const SizeSlider = (props: Props, ref: Ref<Component<'div'>>) => {
  return <Slider marks={SIZE_MARKS} {...props} {...ref} />
}

export default forwardRef(SizeSlider)

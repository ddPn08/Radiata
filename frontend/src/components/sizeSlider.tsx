import { Slider, SliderProps } from '@mantine/core'
import { forwardRef, RefAttributes } from 'react'

// 256 to 1024, 64 steps
export const SIZE_MARKS = Array.from({ length: 16 }, (_, i) => i * 64 + 256).map((v) => ({
  value: v,
  label: null,
}))

type Props = SliderProps & RefAttributes<HTMLDivElement>

const SizeSlider = (props: Props) => {
  return <Slider defaultValue={768} min={256} max={1024} step={64} marks={SIZE_MARKS} {...props} />
}

export default forwardRef(SizeSlider)

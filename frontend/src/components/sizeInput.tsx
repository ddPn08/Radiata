import { Flex, Input, InputWrapperBaseProps, NumberInput } from '@mantine/core'
import { useState } from 'react'
import SizeSlider from './sizeSlider'

// 256 to 1024, 64 steps
export const SIZE_MARKS = Array.from({ length: 16 }, (_, i) => i * 64 + 256).map((v) => ({
  value: v,
  label: null,
}))

interface Props extends InputWrapperBaseProps {
  defaultValue?: number
  min?: number
  max?: number
  step?: number
}

const SizeInput = ({ defaultValue = 768, min = 256, max = 1024, step = 64, ...props }: Props) => {
  const [value, setValue] = useState<number>(defaultValue)

  return (
    <Input.Wrapper {...props}>
      <Flex align={'center'} gap={'sm'}>
        <SizeSlider
          defaultValue={defaultValue}
          min={min}
          max={max}
          step={step}
          w={'100%'}
          onChange={(e) => {
            setValue(e)
          }}
        />
        <NumberInput
          value={value}
          min={min}
          max={max}
          step={step}
          defaultValue={defaultValue}
          w={'100px'}
        />
      </Flex>
    </Input.Wrapper>
  )
}

export default SizeInput

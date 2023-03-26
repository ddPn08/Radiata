import { Flex, Input, InputWrapperBaseProps, Slider } from '@mantine/core'
import { useRef, useState } from 'react'

import { BetterNumInput } from './betterNumInput'

interface Props extends InputWrapperBaseProps {
  defaultValue?: number | undefined
  min?: number | undefined
  max?: number | undefined
  step?: number | undefined
  precision?: number | undefined
  onChange?: ((value: number) => void) | undefined
}

export const NumberSliderInput = ({
  label,
  defaultValue = 768,
  min,
  max,
  step,
  precision,
  onChange,
  ...props
}: Props) => {
  const [value, setValue] = useState<number>(defaultValue)

  const sliderRef = useRef<HTMLInputElement>(null)

  return (
    <Input.Wrapper label={label}>
      <Flex align={'center'} gap={'sm'}>
        <Slider
          label={value}
          defaultValue={defaultValue}
          value={value}
          min={min as number}
          max={max as number}
          step={step as number}
          w={'100%'}
          onChange={(e) => {
            setValue(e)
            onChange && onChange(e)
          }}
          ref={sliderRef}
        />
        <BetterNumInput
          value={value}
          min={min as number}
          max={max as number}
          step={step as number}
          defaultValue={defaultValue}
          precision={precision as number}
          w={'100px'}
          onChange={(e) => {
            if (!e) return

            setValue(e)
            onChange && onChange(e)
          }}
          onWheel={(e) => {
            // blur した際に value が確定され、onChage が発火するため、
            // わざと一度 blur してから focus しなおしている
            e.currentTarget.blur()
            e.currentTarget.focus()
          }}
          {...props}
          allowWheel
        />
      </Flex>
    </Input.Wrapper>
  )
}

import { NumberInput, NumberInputProps } from '@mantine/core'
import { forwardRef, Ref } from 'react'

interface Props extends NumberInputProps {
  allowWheel?: boolean | undefined
}

const BetterNumInput = (
  { min, max, step, allowWheel, onWheel, ...props }: Props,
  ref: Ref<HTMLInputElement>,
) => {
  const updateValueWithWheel = (e: React.WheelEvent<HTMLInputElement>) => {
    if (!step) return

    e.preventDefault()

    const delta = e.deltaY > 0 ? -step : step
    const newValue = Number(e.currentTarget.value) + delta

    if (min && newValue < min) {
      return
    } else if (max && newValue > max) {
      return
    }

    e.currentTarget.value = String(newValue)
  }

  return (
    <NumberInput
      min={min!}
      max={max!}
      step={step!}
      onBlur={(e) => {
        const value = Number(e.currentTarget.value)

        if (isNaN(value)) {
          if (props.defaultValue) {
            e.currentTarget.value = String(props.defaultValue)
          }
          return
        }

        if (!step) {
          return
        }

        if (!min && !max) {
          return
        }

        if (min && max) {
          if ((max - min) / step !== Math.floor((max - min) / step)) {
            console.warn('Must be a factor of max-min')
            return
          }
        }

        if (min) {
          const quotient = Math.floor((value - min) / step)
          if (quotient * step + min !== value) {
            const smaller = quotient * step + min
            const larger = (quotient + 1) * step + min
            const correct = Math.abs(value - smaller) < Math.abs(value - larger) ? smaller : larger
            e.currentTarget.value = String(correct)
            return
          }
        }

        if (max) {
          const quotient = Math.floor((max - value) / step)
          if (quotient * step + value !== max) {
            const smaller = quotient * step + value
            const larger = (quotient + 1) * step + value
            const correct = Math.abs(value - smaller) < Math.abs(value - larger) ? smaller : larger
            e.currentTarget.value = String(correct)
            return
          }
        }
      }}
      onWheel={(e) => {
        if (allowWheel) {
          updateValueWithWheel(e)
        }
        onWheel && onWheel(e)
      }}
      {...props}
      {...ref}
    />
  )
}

export default forwardRef(BetterNumInput)

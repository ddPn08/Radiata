import { Checkbox, Stack } from '@mantine/core'
import type { SetModeRequestModeEnum } from 'internal:api'
import { useEffect, useState } from 'react'

import { api } from '~/api'

export const ModeSelector = () => {
  const [mode, setMode] = useState<SetModeRequestModeEnum>('diffusers')
  const refresh = async () => {
    const { data } = await api.getModelMode()
    setMode(data)
  }
  const change = async (mode: SetModeRequestModeEnum) => {
    setMode(mode)
    await api.setModelMode({ setModeRequest: { mode: mode } })
  }
  useEffect(() => {
    refresh()
  }, [])
  return (
    <Stack>
      <Checkbox
        label="diffusers"
        checked={mode === 'diffusers'}
        onChange={(e) => {
          if (e.currentTarget.checked) change('diffusers')
        }}
      />
      <Checkbox
        label="TensorRT"
        checked={mode === 'tensorrt'}
        onChange={(e) => {
          if (e.currentTarget.checked) change('tensorrt')
        }}
      />
    </Stack>
  )
}

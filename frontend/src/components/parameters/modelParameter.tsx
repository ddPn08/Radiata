import { ActionIcon, Flex, Input, NativeSelect } from '@mantine/core'
import { IconRotateClockwise } from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { api } from '~/api'

const ModelParameter = () => {
  const [models, setModels] = useState<string[]>([])
  const [currentModel, setCurrentModel] = useState<string>('')

  const onModelChange = (value: string) => {}

  const modelRefresh = async () => {
    const runners = await api.getRunners().then((res) => res.data)
    setModels(runners)

    const currentRunner = await api.getCurrentRunner().then((res) => res.data)
    setCurrentModel(currentRunner)
  }

  useEffect(() => {
    modelRefresh()
  }, [])

  return (
    <Input.Wrapper label="Model">
      <Flex align={'center'}>
        <NativeSelect
          data={models}
          w={'100%'}
          onChange={(e) => {
            if (e.target.value) {
              onModelChange(e.target.value)
            }
          }}
        />
        <ActionIcon variant={'outline'} color={'blue'} m={'sm'} onClick={modelRefresh}>
          <IconRotateClockwise size={16} />
        </ActionIcon>
      </Flex>
    </Input.Wrapper>
  )
}

export default ModelParameter

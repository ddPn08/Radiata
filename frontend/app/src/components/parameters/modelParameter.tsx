import { ActionIcon, Flex, Input, NativeSelect, Skeleton } from '@mantine/core'
import { IconRotateClockwise } from '@tabler/icons-react'
import { useEffect, useState } from 'react'

import { api } from '~/api'

const ModelParameter = () => {
  const [models, setModels] = useState<string[]>([])
  const [currentModel, setCurrentModel] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)

  const onModelChange = async (value: string) => {
    if (value === currentModel) {
      return
    }

    if (models.includes(value)) {
      setLoading(true)
      await api.setModel({
        setModelRequest: {
          model_id: value,
        },
      })
      setLoading(false)
      setCurrentModel(value)
    }
  }

  const modelRefresh = async () => {
    const models = await api.getModels().then((res) => res.data!)
    setModels(models.map((v) => v.model_id))

    const currentModel = await api.getCurrentModel().then((res) => res.data!)
    setCurrentModel(currentModel.model_id as string)
  }

  useEffect(() => {
    modelRefresh()
  }, [])

  return (
    <Input.Wrapper label="Model">
      <Skeleton visible={loading}>
        <Flex align={'center'}>
          <NativeSelect
            data={models}
            value={currentModel}
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
      </Skeleton>
    </Input.Wrapper>
  )
}

export default ModelParameter

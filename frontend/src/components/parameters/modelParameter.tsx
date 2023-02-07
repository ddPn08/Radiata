import { ActionIcon, Flex, Input, NativeSelect } from '@mantine/core'
import { IconRotateClockwise } from '@tabler/icons-react'

const ModelParameter = () => {
  return (
    <Input.Wrapper label="Model">
      <Flex align={'center'}>
        <NativeSelect data={['Stable Diffusion v1.5']} w={'100%'} />
        <ActionIcon variant={'outline'} color={'blue'} m={'sm'}>
          <IconRotateClockwise size={16} />
        </ActionIcon>
      </Flex>
    </Input.Wrapper>
  )
}

export default ModelParameter

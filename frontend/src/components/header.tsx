import { ActionIcon, Box, Flex, NativeSelect } from '@mantine/core'
import { IconRotateClockwise } from '@tabler/icons-react'

const Header = () => {
  return (
    <Box mx={'md'} my={'sm'}>
      <Flex align={'center'}>
        <NativeSelect
          data={['Stable Diffusion v1.5']}
          label="Model"
          w={{
            sm: '80%',
            md: '50%',
            lg: '40%',
          }}
        />

        <ActionIcon variant={'light'} m={'md'} top={'10px'}>
          <IconRotateClockwise size={24} />
        </ActionIcon>
      </Flex>
    </Box>
  )
}

export default Header

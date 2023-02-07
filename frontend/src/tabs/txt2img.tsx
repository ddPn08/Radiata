import {
  ActionIcon,
  Box,
  Button,
  Divider,
  Flex,
  Image,
  Input,
  MediaQuery,
  NativeSelect,
  NumberInput,
  Slider,
  Stack,
  Textarea,
} from '@mantine/core'
import { IconArrowsDownUp, IconDice5, IconRotateClockwise } from '@tabler/icons-react'
import SizeInput from '../components/sizeInput'

const Txt2Img = () => {
  return (
    <Box h={'100%'}>
      <Flex h={'100%'}>
        <Stack w={'100%'} m={'md'}>
          <Stack w={'100%'}>
            <Textarea label={'Positive'} autosize />
            <Textarea label={'Negative'} autosize />
          </Stack>

          <Button>Generate</Button>

          <Box>
            <Image />
          </Box>
        </Stack>

        <Divider orientation="vertical" />

        <MediaQuery
          smallerThan={'sm'}
          styles={{
            display: 'none',
          }}
        >
          <Flex
            w={{
              sm: 540,
              md: 640,
              lg: 720,
            }}
            direction={'column'}
            p={'md'}
            gap={'md'}
          >
            <Input.Wrapper label="Model">
              <Flex align={'center'}>
                <NativeSelect data={['Stable Diffusion v1.5']} w={'100%'} />
                <ActionIcon variant={'outline'} color={'blue'} m={'sm'}>
                  <IconRotateClockwise size={16} />
                </ActionIcon>
              </Flex>
            </Input.Wrapper>

            <Flex align={'center'}>
              <Box w={'100%'}>
                {/* Width */}
                <SizeInput label={'Width'} />

                {/* Helight */}
                <SizeInput label={'Height'} />
              </Box>
              <ActionIcon variant="outline" m={'sm'} color={'blue'}>
                <IconArrowsDownUp size={16} />
              </ActionIcon>
            </Flex>

            <NativeSelect label={'Sampler'} data={['euler a']} />
            <Input.Wrapper label={'Seed'}>
              <Flex w={'100%'} align={'center'} gap={'sm'}>
                <NumberInput defaultValue={-1} w={'100%'} />
                <ActionIcon variant={'outline'} color={'blue'}>
                  <IconDice5 size={16} />
                </ActionIcon>
              </Flex>
            </Input.Wrapper>

            <Input.Wrapper label={'Steps'}>
              <Flex align={'center'} gap={'sm'}>
                <Slider defaultValue={50} min={1} max={150} step={1} w={'100%'} />
                <NumberInput defaultValue={50} min={1} max={150} step={1} w={'100px'} />
              </Flex>
            </Input.Wrapper>

            <Input.Wrapper label={'CFG Scale'}>
              <Flex align={'center'} gap={'sm'}>
                <Slider defaultValue={7} min={0.1} max={20} step={0.1} w={'100%'} />
                <NumberInput defaultValue={7} min={0.1} max={20} step={0.1} w={'100px'} />
              </Flex>
            </Input.Wrapper>
          </Flex>
        </MediaQuery>
      </Flex>
    </Box>
  )
}

export default Txt2Img

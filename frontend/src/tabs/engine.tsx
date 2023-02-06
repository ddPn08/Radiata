import {
  Button,
  Checkbox,
  Container,
  Input,
  NativeSelect,
  NumberInput,
  SimpleGrid,
  Space,
  Stack,
  Text,
} from '@mantine/core'
import SizeInput from '../components/sizeInput'

const Engine = () => {
  return (
    <Container py={'md'}>
      <Text size={'lg'}>Build TensorRT from diffusers moodel on Hugging Face</Text>
      <Stack my={'sm'}>
        <Input.Wrapper label={'Model ID'} withAsterisk>
          <Input placeholder="hugging face model id (e.g. CompVis/stablediffusion-v1-4)" />
        </Input.Wrapper>

        <Input.Wrapper label={'Hugging Face Access Token'}>
          <Input placeholder="hf_********************" />
        </Input.Wrapper>

        <SizeInput label={'Optimization Image Width'} />

        <SizeInput label={'Optimization Image Height'} />

        <Input.Wrapper label={'Denoising precision'}>
          <NativeSelect data={['float32', 'float16']} />
        </Input.Wrapper>

        <Input.Wrapper label={'Max batch size'}>
          <NumberInput min={1} max={32} defaultValue={1} />
        </Input.Wrapper>

        <SimpleGrid
          cols={4}
          spacing="lg"
          breakpoints={[
            { maxWidth: 'md', cols: 3, spacing: 'md' },
            { maxWidth: 'sm', cols: 2, spacing: 'sm' },
            { maxWidth: 'xs', cols: 1, spacing: 'sm' },
          ]}
        >
          <Checkbox label={'Build static batch'} />
          <Checkbox label={'Build dynamic shape'} />
          <Checkbox label={'Build preview features'} />
          <Checkbox label={'Force engine build'} />
          <Checkbox label={'Force onnx export'} />
          <Checkbox label={'Force onnx optimize'} />
          <Checkbox label={'Onnx minimal optimization'} />
        </SimpleGrid>

        <Space h={'md'} />

        <Button>Build</Button>
      </Stack>
    </Container>
  )
}

export default Engine

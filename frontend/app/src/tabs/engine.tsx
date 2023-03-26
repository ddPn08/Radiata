import {
  Button,
  Checkbox,
  Container,
  Input,
  NumberInput,
  SimpleGrid,
  Space,
  Stack,
  Text,
  Box,
  Alert,
  Loader,
} from '@mantine/core'
import { IconInfoCircle } from '@tabler/icons-react'
import type { BuildEngineOptions } from 'internal:api'
import { useAtom } from 'jotai'
import { useState } from 'react'

import { NumberSliderInput } from '../components/ui/numberSliderInput'

import { api } from '~/api'
import { buildEngineOptions } from '~/atoms/engine'
import { ModelParameter } from '~/components/parameters/modelParameter'
import { IMAGE_SIZE_STEP, MAX_IMAGE_SIZE, MIN_IMAGE_SIZE } from '~/utils/static'

export const Engine = () => {
  const [form, setForm] = useAtom(buildEngineOptions)

  const [building, setBuilding] = useState(false)
  const [status, setStatus] = useState<'success' | 'error' | null>(null)
  const [error, setError] = useState<string | null>(null)

  const buildEngine = async (req: BuildEngineOptions) => {
    try {
      setStatus(null)
      setBuilding(true)
      await api.buildEngine({ buildEngineOptions: req })
      setBuilding(false)
      setStatus('success')
    } catch (e) {
      setBuilding(false)
      setStatus('error')
      setError((e as Error).message)
    }
  }

  return (
    <Box
      h="100%"
      sx={{
        overflowY: 'auto',
      }}
    >
      <Container py={'md'}>
        <Text size={'lg'}>Build TensorRT from diffusers moodel on Hugging Face</Text>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            buildEngine(form)
          }}
        >
          <Stack my={'sm'}>
            <ModelParameter />

            <Input.Wrapper label={'Hugging Face Access Token'}>
              <Input
                placeholder="hf_********************"
                defaultValue={form.hf_token}
                onChange={(e) =>
                  setForm({
                    ...form,
                    hf_token: e.currentTarget.value,
                  })
                }
              />
            </Input.Wrapper>

            <Input.Wrapper label={'Sub Folder'}>
              <Input
                placeholder=""
                defaultValue={form.subfolder}
                onChange={(e) =>
                  setForm({
                    ...form,
                    subfolder: e.currentTarget.value,
                  })
                }
              />
            </Input.Wrapper>

            <Input.Wrapper label={'Max batch size'}>
              <NumberInput
                min={1}
                max={32}
                defaultValue={form.max_batch_size}
                onChange={(value) => setForm({ ...form, max_batch_size: value || 1 })}
              />
            </Input.Wrapper>

            <NumberSliderInput
              label={'Optimization Image Width'}
              defaultValue={form.opt_image_width}
              min={MIN_IMAGE_SIZE}
              max={MAX_IMAGE_SIZE}
              step={IMAGE_SIZE_STEP}
              onChange={(value) => setForm({ ...form, opt_image_width: value })}
            />

            <NumberSliderInput
              label={'Optimization Image Height'}
              defaultValue={form.opt_image_height}
              min={MIN_IMAGE_SIZE}
              max={MAX_IMAGE_SIZE}
              step={IMAGE_SIZE_STEP}
              onChange={(value) => setForm({ ...form, opt_image_height: value })}
            />

            {form.build_dynamic_shape && (
              <>
                <Input.Wrapper label={'Min latent resolution'}>
                  <NumberInput
                    defaultValue={form.min_latent_resolution}
                    onChange={(value) => setForm({ ...form, min_latent_resolution: value || 256 })}
                  />
                </Input.Wrapper>
                <Input.Wrapper label={'Max latent resolution'}>
                  <NumberInput
                    defaultValue={form.max_latent_resolution}
                    onChange={(value) => setForm({ ...form, max_latent_resolution: value || 1024 })}
                  />
                </Input.Wrapper>
              </>
            )}

            <SimpleGrid
              cols={4}
              spacing="lg"
              breakpoints={[
                { maxWidth: 'md', cols: 3, spacing: 'md' },
                { maxWidth: 'sm', cols: 2, spacing: 'sm' },
                { maxWidth: 'xs', cols: 1, spacing: 'sm' },
              ]}
            >
              <Checkbox
                label={'Build enable refit'}
                defaultChecked={form.build_enable_refit}
                onChange={(e) => setForm({ ...form, build_enable_refit: e.currentTarget.checked })}
              />
              <Checkbox
                label={'Build static batch'}
                defaultChecked={form.build_static_batch}
                onChange={(e) => setForm({ ...form, build_static_batch: e.currentTarget.checked })}
              />
              <Checkbox
                label={'Build dynamic shape'}
                defaultChecked={form.build_dynamic_shape}
                onChange={(e) => setForm({ ...form, build_dynamic_shape: e.currentTarget.checked })}
              />
              <Checkbox
                label={'Build all tactics'}
                defaultChecked={form.build_all_tactics}
                onChange={(e) => setForm({ ...form, build_all_tactics: e.currentTarget.checked })}
              />
              <Checkbox
                label={'Build preview features'}
                defaultChecked={form.build_preview_features}
                onChange={(e) =>
                  setForm({ ...form, build_preview_features: e.currentTarget.checked })
                }
              />
              <Checkbox
                label={'Force engine build'}
                defaultChecked={form.force_engine_build}
                onChange={(e) => setForm({ ...form, force_engine_build: e.currentTarget.checked })}
              />
              <Checkbox
                label={'Force onnx export'}
                defaultChecked={form.force_onnx_export}
                onChange={(e) => setForm({ ...form, force_onnx_export: e.currentTarget.checked })}
              />
              <Checkbox
                label={'Force onnx optimize'}
                defaultChecked={form.force_onnx_optimize}
                onChange={(e) => setForm({ ...form, force_onnx_optimize: e.currentTarget.checked })}
              />
            </SimpleGrid>

            <Space h={'md'} />

            {building ? (
              <Box w={'100%'}>
                <Alert title={'Processing...'}>
                  <Text>
                    This may take about 10 minutes. Please wait until the process is finished.
                  </Text>
                </Alert>
                <Button w={'100%'} my={'sm'} disabled>
                  <Loader p={'xs'} />
                </Button>
              </Box>
            ) : (
              <Button type={'submit'}>Build</Button>
            )}

            {status === 'success' && (
              <Box>
                <Alert
                  title={'Success!'}
                  color={'green'}
                  withCloseButton={true}
                  onClose={() => {
                    setStatus(null)
                  }}
                >
                  <Text>The model has been built successfully. You can now generate images!</Text>
                </Alert>
              </Box>
            )}
          </Stack>
        </form>

        {status === 'error' && (
          <Box>
            <Alert icon={<IconInfoCircle />} title={'Something went wrong...'} color={'red'}>
              {error}
            </Alert>
          </Box>
        )}
      </Container>
    </Box>
  )
}
